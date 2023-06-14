import collections
import json
import os
from PIL import Image
import numpy as np
import time
import tqdm
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset
from torchmetrics import BLEUScore
import torchvision

from gill import losses as losses_utils
from gill import utils
from gill import data


def validate(val_loader, model, tokenizer, criterion, epoch, args):
  ngpus_per_node = torch.cuda.device_count()
  writer = SummaryWriter(args.log_dir)
  bleu_scorers = [BLEUScore(n_gram=i) for i in [1, 2, 3, 4]]
  actual_step = (epoch + 1) * args.steps_per_epoch
  model_modes = ['captioning', 'retrieval', 'generation']
  num_words = 32  # Number of words to generate.

  feature_extractor = utils.get_feature_extractor_for_model(args.visual_model, image_size=args.image_size, train=False)

  def get_pixel_values_from_path(path: str):
    img = Image.open(path)
    img = img.resize((args.image_size, args.image_size))
    pixel_values = utils.get_pixel_values_for_model(feature_extractor, img)[None, ...]

    if args.precision == 'fp16':
        pixel_values = pixel_values.half()
    elif args.precision == 'bf16':
        pixel_values = pixel_values.bfloat16()
    if torch.cuda.is_available():
      pixel_values = pixel_values.cuda()
    return pixel_values

  def run_validate(loader, base_progress=0):
    with torch.no_grad():
      end = time.time()
      all_generated_captions = []
      all_gt_captions = []
      all_generated_image_paths = []
      all_image_features = []
      all_text_features = []

      for i, (image_paths, images, caption_images, ret_tokens, ret_caption_len, gen_tokens, gen_caption_len, clip_emb) in tqdm.tqdm(enumerate(loader), position=0, total=len(loader)):
        i = base_progress + i

        if torch.cuda.is_available():
          ret_tokens = ret_tokens.cuda(args.gpu, non_blocking=True)
          ret_caption_len = ret_caption_len.cuda(args.gpu, non_blocking=True)
          gen_tokens = gen_tokens.cuda(args.gpu, non_blocking=True)
          gen_caption_len = gen_caption_len.cuda(args.gpu, non_blocking=True)
          images = images.cuda()
          clip_emb = clip_emb.cuda()

        if args.precision == 'fp16':
          images = images.half()
        elif args.precision == 'bf16':
          images = images.bfloat16()

        for model_mode in model_modes:
          # compute output
          if model_mode == 'retrieval':
            tgt_tokens, token_len = ret_tokens, ret_caption_len
          elif model_mode == 'generation':
            tgt_tokens, token_len = gen_tokens, gen_caption_len
          else:
            tgt_tokens, token_len = ret_tokens, ret_caption_len  # For captioning, it doesn't matter.

          (model_output, full_labels, last_embedding, _, visual_embs, visual_embs_norm,
            input_embs_norm, _) = model(images, tgt_tokens, token_len, mode=model_mode, input_prefix=args.input_prompt)  # (N, T, C)

          if model_mode == 'captioning':
            loss = args.cap_loss_scale * model_output.loss
          elif model_mode == 'retrieval':
            loss = args.ret_loss_scale * model_output.loss
          elif model_mode == 'generation':
            loss = args.ret_loss_scale * model_output.loss
          else:
            raise NotImplementedError

          output = model_output.logits
          if model_mode == 'captioning':
            acc1, acc5 = utils.accuracy(output[:, :-1, :], full_labels[:, 1:], -100, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            ce_losses.update(loss.item(), images.size(0))
          elif model_mode == 'retrieval':
            if args.distributed:
              original_last_embedding = torch.clone(last_embedding)
              all_visual_embs = [torch.zeros_like(visual_embs) for _ in range(dist.get_world_size())]
              all_last_embedding = [torch.zeros_like(last_embedding) for _ in range(dist.get_world_size())]

              dist.all_gather(all_visual_embs, visual_embs)
              dist.all_gather(all_last_embedding, last_embedding)

              # Overwrite with embeddings produced on this replica, which track the gradients.
              all_visual_embs[dist.get_rank()] = visual_embs
              all_last_embedding[dist.get_rank()] = last_embedding
              visual_embs = torch.cat(all_visual_embs)
              last_embedding = torch.cat(all_last_embedding)
              start_idx = args.rank * images.shape[0]
              end_idx = start_idx + images.shape[0]
              assert torch.all(last_embedding[start_idx:end_idx] == original_last_embedding), args.rank

            all_text_features.append(last_embedding.cpu())
            all_image_features.append(visual_embs.cpu())
          elif model_mode == 'generation':
            if args.num_clip_tokens != args.num_tokens:
              seq_len = clip_emb.shape[1]
              last_embedding = last_embedding.reshape((last_embedding.shape[0], seq_len, -1))
              assert last_embedding.shape == clip_emb.shape, (last_embedding.shape == clip_emb.shape)
            image_loss = losses_utils.l2_loss(clip_emb, last_embedding)  # (N,)
            gen_loss = args.gen_loss_scale * image_loss.mean()
            gen_losses.update(gen_loss.item(), image_loss.size(0))

          # Run auto-regressive generation sample
          if model_mode == 'captioning':
            min_word_tokens = num_words

            input_embs = model.module.model.get_visual_embs(images, mode='captioning')  # (2, n_visual_tokens, D)
            if args.input_prompt is not None:
              print(f'Adding prefix "{args.input_prompt}" to captioning generate=True.')
              prompt_ids = tokenizer(args.input_prompt, add_special_tokens=True, return_tensors="pt").input_ids
              prompt_ids = prompt_ids.to(visual_embs.device)
              prompt_embs = model.module.model.input_embeddings(prompt_ids)
              prompt_embs = prompt_embs.repeat(input_embs.shape[0], 1, 1)
              input_embs = torch.cat([input_embs, prompt_embs], dim=1)

            generated_ids, _, _ = model(input_embs, tgt_tokens, token_len,
              generate=True, num_words=num_words, temperature=0.0, top_p=1.0,
              min_word_tokens=min_word_tokens)

            if args.distributed and ngpus_per_node > 1:
              all_generated_ids = [torch.zeros_like(generated_ids) for _ in range(dist.get_world_size())]
              dist.all_gather(all_generated_ids, generated_ids)
              all_generated_ids[dist.get_rank()] = generated_ids
              generated_ids = torch.cat(all_generated_ids)

              all_tgt_tokens = [torch.zeros_like(tgt_tokens) for _ in range(dist.get_world_size())]
              dist.all_gather(all_tgt_tokens, tgt_tokens)
              all_tgt_tokens[dist.get_rank()] = tgt_tokens
              all_tgt_tokens = torch.cat(all_tgt_tokens)

              all_image_paths = [[None for _ in image_paths] for _ in range(dist.get_world_size())]
              dist.all_gather_object(all_image_paths, image_paths)
              all_image_paths[dist.get_rank()] = image_paths
              image_paths = []
              for p in all_image_paths:
                image_paths.extend(p)
            else:
              all_tgt_tokens = tgt_tokens

            all_tgt_tokens[all_tgt_tokens == -100] = tokenizer.pad_token_id
            generated_captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            gt_captions = tokenizer.batch_decode(all_tgt_tokens, skip_special_tokens=True)

            for cap_i in range(len(generated_captions)):
              image_path = image_paths[cap_i]
              all_generated_image_paths.append(image_path)
              stop_idx = generated_captions[cap_i].find('.')
              if stop_idx > 5:
                all_generated_captions.append(generated_captions[cap_i][:stop_idx])
              else:
                all_generated_captions.append(generated_captions[cap_i])
              all_gt_captions.append([gt_captions[cap_i]])
          elif model_mode in ['retrieval', 'generation']:
            if i == 0:
              # Generate without conditions just to test.
              input_ids = tgt_tokens[:, :3]  # Use first 3 tokens as initial prompt for generation.
              input_embs = model.module.model.input_embeddings(input_ids)  # (N, T, D)
              generated_ids, _, _ = model(input_embs, tgt_tokens, token_len, generate=True, num_words=num_words, temperature=0.0, top_p=1.0)
              generated_ids = torch.cat([input_ids, generated_ids], dim=1)
              generated_captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
              gt_captions = tokenizer.batch_decode(tgt_tokens, skip_special_tokens=False)
          else:
            raise NotImplementedError

          if i == 0:
            max_to_display = 5
            print('=' * 30)
            print('Generated samples:')
            for cap_i, cap in enumerate(generated_captions[:max_to_display]):
              print(f'{cap_i}) {cap}')
            print('=' * 30)
            print('Real samples:')
            for cap_i, cap in enumerate(gt_captions[:max_to_display]):
              print(f'{cap_i}) {cap}')
            print('=' * 30)

            # Write images.
            if not args.distributed or (args.rank % ngpus_per_node == 0):
              max_images_to_show = 16
              normalized_images = images - images.min()
              normalized_images /= normalized_images.max()  # (N, 3, H, W)
              # Create generated caption text.
              generated_cap_images = torch.stack([
                utils.create_image_of_text(
                  generated_captions[j].encode('ascii', 'ignore'),
                  width=normalized_images.shape[3],
                  color=(255, 255, 0))
                for j in range(normalized_images.shape[0])], axis=0)
              # Append gt/generated caption images.
              display_images = torch.cat([normalized_images.float().cpu(), caption_images, generated_cap_images], axis=2)[:max_images_to_show]
              grid = torchvision.utils.make_grid(display_images, nrow=int(max_images_to_show ** 0.5), padding=4)
              writer.add_image(f'val/images_{model_mode}', grid, actual_step)

          vis_emb_norm.update(visual_embs_norm.item(), images.size(0))
          inp_emb_norm.update(input_embs_norm.item(), images.size(0))

          # measure elapsed time
          batch_time.update(time.time() - end)
          end = time.time()

        if i % args.print_freq == 0:
          progress.display(i + 1)

        if i == args.val_steps_per_epoch - 1:
          break

      # Measure captioning metrics.
      path2captions = collections.defaultdict(list)
      for image_path, caption in zip(all_generated_image_paths, all_gt_captions):
        assert len(caption) == 1, caption
        trunc_cap = caption[0]
        for i in range(args.num_tokens):
          trunc_cap = trunc_cap.replace(f'[IMG{i}]', '')
        path2captions[image_path].append(trunc_cap.strip())
      full_gt_captions = [path2captions[path] for path in all_generated_image_paths]

      print(f'Computing BLEU with {len(all_generated_captions)} generated captions:'
            f'{all_generated_captions[:5]} and {len(full_gt_captions)} groundtruth captions:',
            f'{full_gt_captions[:5]}.')
      bleu1_score = bleu_scorers[0](all_generated_captions, full_gt_captions)
      bleu1.update(bleu1_score, 1)
      bleu2_score = bleu_scorers[1](all_generated_captions, full_gt_captions)
      bleu2.update(bleu2_score, 1)
      bleu3_score = bleu_scorers[2](all_generated_captions, full_gt_captions)
      bleu3.update(bleu3_score, 2)
      bleu4_score = bleu_scorers[3](all_generated_captions, full_gt_captions)
      bleu4.update(bleu4_score, 3)

      # Measure retrieval metrics over the entire validation set.
      all_image_features = torch.cat(all_image_features, axis=0)  # (coco_val_len, 2048)
      all_text_features = torch.cat(all_text_features, axis=0)  # (coco_val_len, 2048)

      print(f"Computing similarity between {all_image_features.shape} and {all_text_features.shape}.")
      logits_per_image = all_image_features @ all_text_features.t()
      logits_per_text = logits_per_image.t()
      all_image_acc1, all_image_acc5 = losses_utils.contrastive_acc(logits_per_image, topk=(1, 5))
      all_caption_acc1, all_caption_acc5 = losses_utils.contrastive_acc(logits_per_text, topk=(1, 5))
      image_loss = losses_utils.contrastive_loss(logits_per_image)
      caption_loss = losses_utils.contrastive_loss(logits_per_text)

      loss = args.ret_loss_scale * (image_loss + caption_loss) / 2.0
      cont_losses.update(loss.item(), logits_per_image.size(0))
      top1_caption.update(all_caption_acc1.item(), logits_per_image.size(0))
      top5_caption.update(all_caption_acc5.item(), logits_per_image.size(0))
      top1_image.update(all_image_acc1.item(), logits_per_image.size(0))
      top5_image.update(all_image_acc5.item(), logits_per_image.size(0))


  batch_time = utils.AverageMeter('Time', ':6.3f', utils.Summary.AVERAGE)
  cont_losses = utils.AverageMeter('ContLoss', ':.4e', utils.Summary.AVERAGE)
  ce_losses = utils.AverageMeter('CeLoss', ':.4e', utils.Summary.AVERAGE)
  gen_losses = utils.AverageMeter('GenLoss', ':.4e', utils.Summary.AVERAGE)
  top1 = utils.AverageMeter('Acc@1', ':6.2f', utils.Summary.AVERAGE)
  top5 = utils.AverageMeter('Acc@5', ':6.2f', utils.Summary.AVERAGE)
  bleu1 = utils.AverageMeter('BLEU@1', ':6.2f', utils.Summary.AVERAGE)
  bleu2 = utils.AverageMeter('BLEU@2', ':6.2f', utils.Summary.AVERAGE)
  bleu3 = utils.AverageMeter('BLEU@3', ':6.2f', utils.Summary.AVERAGE)
  bleu4 = utils.AverageMeter('BLEU@4', ':6.2f', utils.Summary.AVERAGE)
  vis_emb_norm = utils.AverageMeter('VisualEmbNorm', ':.4e', utils.Summary.AVERAGE)
  inp_emb_norm = utils.AverageMeter('TextEmbNorm', ':.4e', utils.Summary.AVERAGE)
  top1_caption = utils.AverageMeter('CaptionAcc@1', ':6.2f', utils.Summary.AVERAGE)
  top5_caption = utils.AverageMeter('CaptionAcc@5', ':6.2f', utils.Summary.AVERAGE)
  top1_image = utils.AverageMeter('ImageAcc@1', ':6.2f', utils.Summary.AVERAGE)
  top5_image = utils.AverageMeter('ImageAcc@5', ':6.2f', utils.Summary.AVERAGE)

  progress = utils.ProgressMeter(
    len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
    [batch_time, cont_losses, ce_losses, gen_losses, top1, top5, bleu4],
    prefix='Test: ')

  # switch to evaluate mode
  model.eval()

  run_validate(val_loader)
  if args.distributed:
    batch_time.all_reduce()
    cont_losses.all_reduce()
    gen_losses.all_reduce()
    vis_emb_norm.all_reduce()
    inp_emb_norm.all_reduce()
    bleu1.all_reduce()
    bleu2.all_reduce()
    bleu3.all_reduce()
    bleu4.all_reduce()
    top1.all_reduce()
    top5.all_reduce()
    top1_caption.all_reduce()
    top5_caption.all_reduce()
    top1_image.all_reduce()
    top5_image.all_reduce()

  if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
    aux_val_dataset = Subset(val_loader.dataset,
                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
    aux_val_loader = torch.utils.data.DataLoader(
      aux_val_dataset, batch_size=(args.val_batch_size or args.batch_size), shuffle=False,
      num_workers=args.workers, pin_memory=True, collate_fn=data.collate_fn)
    run_validate(aux_val_loader, len(val_loader))

  progress.display_summary()

  writer.add_scalar('val/vis_emb_norm', vis_emb_norm.avg, actual_step)
  writer.add_scalar('val/text_emb_norm', inp_emb_norm.avg, actual_step)
  writer.add_scalar('val/total_secs_per_batch', batch_time.avg, actual_step)
  writer.add_scalar('val/seq_top1_acc', top1.avg, actual_step)
  writer.add_scalar('val/seq_top5_acc', top5.avg, actual_step)
  writer.add_scalar('val/ce_loss', ce_losses.avg, actual_step)
  writer.add_scalar('val/bleu1', bleu1.avg, actual_step)
  writer.add_scalar('val/bleu2', bleu2.avg, actual_step)
  writer.add_scalar('val/bleu3', bleu3.avg, actual_step)
  writer.add_scalar('val/bleu4', bleu4.avg, actual_step)
  writer.add_scalar('val/contrastive_loss', cont_losses.avg, actual_step)
  writer.add_scalar('val/gen_l2_loss', gen_losses.avg, actual_step)
  writer.add_scalar('val/t2i_top1_acc', top1_caption.avg, actual_step)
  writer.add_scalar('val/t2i_top5_acc', top5_caption.avg, actual_step)
  writer.add_scalar('val/i2t_top1_acc', top1_image.avg, actual_step)
  writer.add_scalar('val/i2t_top5_acc', top5_image.avg, actual_step)
  writer.add_scalar('val/top1_acc', (top1_caption.avg + top1_image.avg) / 2.0, actual_step)
  writer.add_scalar('val/top5_acc', (top5_caption.avg + top5_image.avg) / 2.0, actual_step)

  writer.close()

  # Use top1 accuracy as the metric for keeping the best checkpoint.
  return top1_caption.avg
