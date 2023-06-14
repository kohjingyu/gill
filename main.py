"""Training example.

Example usage:

python -u main.py \
    --dataset=cc3m  --val-dataset=cc3m \
    --opt-version='facebook/opt-6.7b' --visual-model='openai/clip-vit-large-patch14' \
    --exp_name='fromage_exp'   --log-base-dir='runs/' \
    --batch-size=64  --val-batch-size=64  --precision='bf16'

Example run on 2 A6000 GPUs to reproduce the model in the paper:

"""
import argparse
from collections import OrderedDict
import json
import os
import random
import sys
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
from warmup_scheduler import GradualWarmupScheduler
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
import torchvision
from transformers import AutoTokenizer

from gill import data
from gill import losses as losses_utils
from gill import models
from gill import utils
from gill import validate


llm_models = ['facebook/opt-125m', 'facebook/opt-350m', 'facebook/opt-1.3b', 'facebook/opt-2.7b',
              'facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b']
datasets = ['cc3m']
best_acc1 = 0  # Variable to keep track of best model so far.


def parse_args(args):
  parser = argparse.ArgumentParser(description='GILL training')
  parser.add_argument('--opt-version', default='facebook/opt-6.7b',
                      choices=llm_models,
                      help='OPT versions: ' +
                        ' | '.join(llm_models) +
                        ' (default: "facebook/opt-6.7b")')
  parser.add_argument('--visual-model', default='openai/clip-vit-large-patch14', type=str,
                      help="Visual encoder to use.")
  parser.add_argument('--num-tokens', default=8, type=int, metavar='N', help='Number of [IMG] tokens to use.')
  parser.add_argument('--num-clip-tokens', default=77, type=int, metavar='N', help='Number of CLIP token to use for generation.')

  parser.add_argument('-d', '--dataset', metavar='DATASET',  help='Delimited list of datasets:' +
                      ' | '.join(datasets), default='cc3.1m',
                      type=lambda s: [x for x in s.split(',')])

  parser.add_argument('--val-dataset', metavar='DATASET', default='cc3.1m',
            type=lambda s: [x for x in s.split(',')],
            help='Validation dataset: ' +
              ' | '.join(datasets) +
              ' (default: cc3.1m)')
  parser.add_argument('--dataset-dir', default='datasets', type=str,
            help='Dataset directory containing .tsv files.')
  parser.add_argument('--image-dir', default='data/', type=str,
            help='Dataset directory containing image folders.')
  parser.add_argument('--log-base-dir', default='./runs', type=str,
            help='Base directory to write logs and ckpts to.')
  parser.add_argument('--exp-name', default='frozen', type=str,
            help='Name of experiment, used for saving checkpoints.')

  parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
            help='number of data loading workers (default: 4)')
  parser.add_argument('--epochs', default=90, type=int, metavar='N',
            help='number of total epochs to run')
  parser.add_argument('--steps_per_epoch', default=2000, type=int, metavar='N',
            help='number of training steps per epoch')
  parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
            help='manual epoch number (useful on restarts)')
  parser.add_argument('--val_steps_per_epoch', default=-1, type=int, metavar='N',
            help='number of validation steps per epoch')
  parser.add_argument('-b', '--batch-size', default=200, type=int,
            metavar='N',
            help='mini-batch size (default: 200), this is the total '
               'batch size of all GPUs on the current node when '
               'using Data Parallel or Distributed Data Parallel')
  parser.add_argument('--val-batch-size', default=None, type=int)
  parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
            metavar='LR', help='initial learning rate', dest='lr')
  parser.add_argument('--lr-warmup-steps', default=2000, type=int,
            metavar='N', help='Number of steps to warm up lr.')
  parser.add_argument('--lr_schedule_step_size', default=5, type=int,
            metavar='N', help='Number of steps before decaying lr.')
  parser.add_argument('--lr_schedule_gamma', default=0.1, type=float,
            metavar='N', help='Decay parameter for learning rate scheduler.')
  parser.add_argument('--grad-accumulation-steps', default=1, type=int, metavar='N',
                      help='number of gradient accumulation steps')
  parser.add_argument('--grad-clip', default=1.0, type=float, help='gradient clipping amount')

  parser.add_argument('--precision', default='bf16', type=str, choices=['fp32', 'fp16', 'bf16'],
                      help="What precision to train in.")
  parser.add_argument('--cap-loss-scale', type=float, default=1.0, help="Scale on captioning loss.")
  parser.add_argument('--ret-loss-scale', type=float, default=1.0, help="Scale on retrieval loss.")
  parser.add_argument('--gen-loss-scale', type=float, default=1.0, help="Scale on retrieval loss.")

  parser.add_argument('--concat-captions-prob', type=float, default=0.5, help="Probability of concatenating two examples sequentially for captioning.")
  parser.add_argument('--input-prompt', default='A picture of', type=str, help="Input prompt for the language model, if any.")

  parser.add_argument('--image-size', default=224, type=int, metavar='N', help='Size of images.')
  parser.add_argument('--ret-emb-dim', default=256, type=int, metavar='N', help='Embedding dimension for retrieval.')
  parser.add_argument('--gen-emb-dim', default=768, type=int, metavar='N', help='Embedding dimension for generation.')
  
  text_fc_modes = ['linear', 'gill_mapper']
  parser.add_argument('--text-fc-mode', default='gill_mapper',
            choices=text_fc_modes, help='What kind of translation mapping to use.')
  parser.add_argument('--ret-text-fc-mode', default='linear',
            choices=text_fc_modes, help='What kind of translation mapping to use.')

  parser.add_argument('--max-len', default=32, type=int,
            metavar='N', help='Maximum length to truncate captions / generations to.')
  parser.add_argument('--n-visual-tokens', default=4, type=int,
            metavar='N', help='Number of visual tokens to use for the Frozen model.')


  parser.add_argument('--beta1', default=0.9, type=float, metavar='M',
            help='beta1 for Adam')
  parser.add_argument('--beta2', default=0.95, type=float, metavar='M',
            help='beta2 for Adam')
  parser.add_argument('--wd', '--weight-decay', default=0.01, type=float,
            metavar='W', help='weight decay (default: 0.01)',
            dest='weight_decay')
  parser.add_argument('-p', '--print-freq', default=10, type=int,
            metavar='N', help='print frequency (default: 10)')
  parser.add_argument('--resume', default='', type=str, metavar='PATH',
            help='path to latest checkpoint (default: none)')
  parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
            help='evaluate model on validation set')
  parser.add_argument('--world-size', default=-1, type=int,
            help='number of nodes for distributed training')
  parser.add_argument('--rank', default=-1, type=int,
            help='node rank for distributed training')
  parser.add_argument('--dist-url', default='tcp://127.0.0.1:1337', type=str,
            help='url used to set up distributed training')
  parser.add_argument('--dist-backend', default='nccl', type=str,
            help='distributed backend')
  parser.add_argument('--seed', default=None, type=int,
            help='seed for initializing training. ')
  parser.add_argument('--gpu', default=None, type=int,
            help='GPU id to use.')
  parser.add_argument('--multiprocessing-distributed', action='store_true',
            help='Use multi-processing distributed training to launch '
               'N processes per node, which has N GPUs. This is the '
               'fastest way to use PyTorch for either single node or '
               'multi node data parallel training')
  return parser.parse_args(args)


def main(args):
  args = parse_args(args)
  i = 1
  args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
  while os.path.exists(args.log_dir):
    args.log_dir = os.path.join(args.log_base_dir, f'{args.exp_name}_{i}')
    i += 1
  os.makedirs(args.log_dir)

  with open(os.path.join(args.log_dir, f'args.json'), 'w') as wf:
    json.dump(vars(args), wf, indent=4)

  with open(os.path.join(args.log_dir, f'git_info.txt'), 'w') as wf:
    utils.dump_git_status(out_file=wf)

  print(f'Logging to {args.log_dir}.')

  if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'You may see unexpected behavior when restarting '
            'from checkpoints.')

  if args.gpu is not None:
    warnings.warn('You have chosen a specific GPU. This will completely '
            'disable data parallelism.')

  if args.dist_url == "env://" and args.world_size == -1:
    args.world_size = int(os.environ["WORLD_SIZE"])

  args.distributed = args.world_size > 1 or args.multiprocessing_distributed

  ngpus_per_node = torch.cuda.device_count()
  if args.multiprocessing_distributed:
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    args.world_size = ngpus_per_node * args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
  else:
    # Simply call main_worker function
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
  global best_acc1
  args.gpu = gpu

  if args.gpu is not None:
    print("Use GPU: {} for training".format(args.gpu))

  if args.distributed:
    if args.dist_url == "env://" and args.rank == -1:
      args.rank = int(os.environ["RANK"])
    if args.multiprocessing_distributed:
      # For multiprocessing distributed training, rank needs to be the
      # global rank among all the processes
      args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                world_size=args.world_size, rank=args.rank)

  # Create model
  model_args = models.GILLArgs()
  model_args.opt_version = args.opt_version
  model_args.visual_encoder = args.visual_model
  model_args.freeze_lm = True
  model_args.freeze_vm = True
  model_args.n_visual_tokens = args.n_visual_tokens
  model_args.ret_emb_dim = args.ret_emb_dim
  model_args.gen_emb_dim = args.gen_emb_dim
  model_args.text_fc_mode = args.text_fc_mode
  model_args.ret_text_fc_mode = args.ret_text_fc_mode
  model_args.num_tokens = args.num_tokens
  model_args.num_clip_tokens = args.num_clip_tokens
  assert args.num_tokens == 0 or 'gill_mapper' in model_args.text_fc_mode or (args.num_tokens * args.gen_emb_dim == args.num_clip_tokens * 768 or args.num_tokens * args.gen_emb_dim == args.num_clip_tokens * 1024), (f'{args.num_tokens} * {args.gen_emb_dim} != {args.num_clip_tokens} * 768 (or 1024)')

  tokenizer = AutoTokenizer.from_pretrained(args.opt_version, use_fast=False)
  if tokenizer.pad_token is None:
    if args.opt_version in ['EleutherAI/gpt-j-6B']:
      tokenizer.pad_token = tokenizer.eos_token
    else:
      tokenizer.pad_token_id = tokenizer.eos_token_id
    print("tokenizer.pad_token, tokenizer.eos_token:", tokenizer.pad_token, tokenizer.eos_token)
  # Add an image token for loss masking (and visualization) purposes.
  tokenizer.add_special_tokens({"cls_token": "<|image|>"})  # add special image token to tokenizer

  # Add [IMG] tokens to the vocabulary.
  model_args.retrieval_token_idx = []
  args.retrieval_token_idx = []
  for i in range(model_args.num_tokens):
    print(f'Adding [IMG{i}] token to vocabulary.')
    print(f'Before adding new token, tokenizer("[IMG{i}]") =', tokenizer(f'[IMG{i}]', add_special_tokens=False))
    num_added_tokens = tokenizer.add_tokens(f'[IMG{i}]')
    print(f'After adding {num_added_tokens} new tokens, tokenizer("[IMG{i}]") =', tokenizer(f'[IMG{i}]', add_special_tokens=False))
    ret_token_idx = tokenizer(f'[IMG{i}]', add_special_tokens=False).input_ids
    assert len(ret_token_idx) == 1, ret_token_idx
    model_args.retrieval_token_idx.append(ret_token_idx[0])
    args.retrieval_token_idx.append(ret_token_idx[0])

  # Add [GEN] tokens to the vocabulary.
  model_args.gen_token_idx = model_args.retrieval_token_idx
  args.gen_token_idx = args.retrieval_token_idx

  # Save model args to disk.
  with open(os.path.join(args.log_dir, 'model_args.json'), 'w') as f:
    json.dump(vars(model_args), f, indent=4)

  model = models.GILL(tokenizer, model_args)
  if args.precision == 'fp16':
    model = model.float()
  elif args.precision == 'bf16':
    model = model.bfloat16()

  # Print parameters and count of model.
  param_counts_text = utils.get_params_count_str(model)
  with open(os.path.join(args.log_dir, 'param_count.txt'), 'w') as f:
    f.write(param_counts_text)

  # Log trainable parameters to Tensorboard.
  _, total_trainable_params, total_nontrainable_params = utils.get_params_count(model)
  writer = SummaryWriter(args.log_dir)
  writer.add_scalar('params/total', total_trainable_params + total_nontrainable_params, 0)
  writer.add_scalar('params/total_trainable', total_trainable_params, 0)
  writer.add_scalar('params/total_non_trainable', total_nontrainable_params, 0)
  writer.close()

  if not torch.cuda.is_available():
    print('WARNING: using CPU, this will be slow!')
    model = torch.nn.DataParallel(model)
  elif args.distributed:
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    if args.gpu is not None:
      torch.cuda.set_device(args.gpu)
      model.cuda(args.gpu)
      # When using a single GPU per process and per
      # DistributedDataParallel, we need to divide the batch size
      # ourselves based on the total number of GPUs of the current node.
      args.batch_size = int(args.batch_size / ngpus_per_node)
      args.val_batch_size = int((args.val_batch_size or args.batch_size) / ngpus_per_node)
      args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
      model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
    else:
      model.cuda()
      # DistributedDataParallel will divide and allocate batch_size to all
      # available GPUs if device_ids are not set
      model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
  elif args.gpu is not None:
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
  else:
    model = torch.nn.DataParallel(model).cuda()

  # define loss function (criterion), optimizer, and learning rate scheduler
  criterion = nn.CrossEntropyLoss().cuda(args.gpu)
  optimizer_cls = torch.optim.AdamW
  print('Using torch.optim.AdamW as the optimizer.')
  optimizer = optimizer_cls(model.parameters(), args.lr,
                betas=(args.beta1, args.beta2),
                weight_decay=args.weight_decay,
                eps=1e-8)

  """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
  scheduler_steplr = StepLR(optimizer, step_size=args.lr_schedule_step_size * args.steps_per_epoch, gamma=args.lr_schedule_gamma)
  scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=args.lr_warmup_steps, after_scheduler=scheduler_steplr)
  
  # optionally resume from a checkpoint
  if args.resume:
    if os.path.isfile(args.resume):
      print("=> loading checkpoint '{}'".format(args.resume))
      if args.gpu is None:
        checkpoint = torch.load(args.resume)
      else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.resume, map_location=loc)
      args.start_epoch = checkpoint['epoch']
      best_acc1 = checkpoint['best_acc1']
      if args.gpu is not None:
        # best_acc1 may be from a checkpoint from a different GPU
        best_acc1 = best_acc1.to(args.gpu)
      model.load_state_dict(checkpoint['state_dict'], strict=False)
      optimizer.load_state_dict(checkpoint['optimizer'])
      scheduler.load_state_dict(checkpoint['scheduler'])
      print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.resume, checkpoint['epoch']))
    else:
      print("=> no checkpoint found at '{}'".format(args.resume))

  cudnn.benchmark = True

  # Data loading code
  train_dataset = data.get_dataset(args, 'train', tokenizer)
  val_dataset = data.get_dataset(args, 'val', tokenizer)
  print(f'Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples.')

  if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
  else:
    train_sampler = None
    val_sampler = None

  train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

  val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=(args.val_batch_size or args.batch_size), shuffle=False,
    num_workers=args.workers, pin_memory=True, sampler=val_sampler)

  if args.evaluate:
    validate.validate(val_loader, model, tokenizer, criterion, epoch, args)
    return

  for epoch in range(args.start_epoch, args.epochs):
    if epoch == 0:
      validate.validate(val_loader, model, tokenizer, criterion, epoch-1, args)
    if args.distributed:
      train_sampler.set_epoch(epoch)

    # train for one epoch
    train(train_loader, model, tokenizer, criterion, optimizer, epoch, scheduler, args)

    # evaluate on validation set
    acc1 = validate.validate(val_loader, model, tokenizer, criterion, epoch, args)

    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        and args.rank % ngpus_per_node == 0):

      # Only save non-frozen parameters.
      stripped_state_dict = {
          k: v for k, v in model.state_dict().items() if 
          ('.lm' not in k and '.visual_model' not in k)
      }
      stripped_state_dict = OrderedDict(sorted(stripped_state_dict.items()))
      utils.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': stripped_state_dict,
        'best_acc1': best_acc1,
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict()
      }, is_best, os.path.join(args.log_dir, 'ckpt'))


def train(train_loader, model, tokenizer, criterion, optimizer, epoch, scheduler, args):
  ngpus_per_node = torch.cuda.device_count()
  batch_time = utils.AverageMeter('Time', ':6.3f')
  cap_time = utils.AverageMeter('CaptioningTime', ':6.3f')
  ret_time = utils.AverageMeter('RetrievalTime', ':6.3f')
  data_time = utils.AverageMeter('Data', ':6.3f')
  losses = utils.AverageMeter('Loss', ':.4e')
  ce_losses = utils.AverageMeter('CeLoss', ':.4e')
  top1 = utils.AverageMeter('Acc@1', ':6.2f')
  top5 = utils.AverageMeter('Acc@5', ':6.2f')
  cont_losses = utils.AverageMeter('ContLoss', ':.4e')
  gen_losses = utils.AverageMeter('GenLoss', ':.4e')
  top1_caption = utils.AverageMeter('AccCaption@1', ':6.2f')
  top5_caption = utils.AverageMeter('AccCaption@5', ':6.2f')
  top1_image = utils.AverageMeter('AccImage@1', ':6.2f')
  top5_image = utils.AverageMeter('AccImage@5', ':6.2f')
  cap_vis_emb_norm = utils.AverageMeter('VisualEmbNormCap', ':.4e')
  ret_vis_emb_norm = utils.AverageMeter('VisualEmbNormRet', ':.4e')
  inp_emb_norm = utils.AverageMeter('TextEmbNorm', ':.4e')
  all_emb_norm = utils.AverageMeter('AllEmbNorm', ':.4e')
  ret_emb_norm = utils.AverageMeter('RetEmbNorm', ':.4e')

  writer = SummaryWriter(args.log_dir)

  progress = utils.ProgressMeter(
    args.steps_per_epoch,
    [batch_time, losses, ce_losses, cont_losses, gen_losses, top1, top5],
    prefix="Epoch: [{}]".format(epoch))

  # switch to train mode
  model.train()
  end = time.time()

  for i, (_, images, caption_images, ret_tokens, ret_caption_len, gen_tokens, gen_caption_len, clip_emb) in enumerate(train_loader):
    actual_step = epoch * args.steps_per_epoch + i + 1
    # measure data loading time
    data_time.update(time.time() - end)

    if torch.cuda.is_available():
      images = images.cuda(args.gpu, non_blocking=True)
      ret_tokens = ret_tokens.cuda(args.gpu, non_blocking=True)
      ret_caption_len = ret_caption_len.cuda(args.gpu, non_blocking=True)
      gen_tokens = gen_tokens.cuda(args.gpu, non_blocking=True)
      gen_caption_len = gen_caption_len.cuda(args.gpu, non_blocking=True)
      clip_emb = clip_emb.cuda(args.gpu, non_blocking=True)

    if args.precision == 'fp16':
      images = images.half()
    elif args.precision == 'bf16':
      images = images.bfloat16()

    model_modes = ['captioning', 'retrieval', 'generation']

    loss = 0

    for model_mode in model_modes:
      print('Running', model_mode)
      mode_start = time.time()
      # compute output
      concat_captions = random.uniform(0, 1) < args.concat_captions_prob

      if model_mode == 'retrieval':
        tgt_tokens, token_len = ret_tokens, ret_caption_len
      elif model_mode == 'generation':
        tgt_tokens, token_len = gen_tokens, gen_caption_len
      else:
        tgt_tokens, token_len = ret_tokens, ret_caption_len  # For captioning, it doesn't matter.

      (model_output, full_labels, last_embedding, _, visual_embs, visual_embs_norm,
        input_embs_norm, _) = model(images, tgt_tokens, token_len, mode=model_mode,
                                 concat_captions=concat_captions)
      output = model_output.logits

      # Measure captioning accuracy for multi-task models and next-token prediction for retrieval models.
      if model_mode == 'captioning':
        acc1, acc5 = utils.accuracy(output[:, :-1, :], full_labels[:, 1:], -100, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

      ce_loss = model_output.loss
      if model_mode == 'captioning':
        ce_loss = ce_loss * args.cap_loss_scale
      elif model_mode == 'retrieval':
        ce_loss = ce_loss * args.ret_loss_scale * 0.5
      elif model_mode == 'generation':
        ce_loss = ce_loss * args.gen_loss_scale * 0.5
      else:
        raise NotImplementedError
      loss += ce_loss
      ce_losses.update(ce_loss.item(), images.size(0))

      if model_mode == 'retrieval':
        # Cross replica concat for embeddings.
        if args.distributed:
          all_visual_embs = [torch.zeros_like(visual_embs) for _ in range(dist.get_world_size())]
          all_last_embedding = [torch.zeros_like(last_embedding) for _ in range(dist.get_world_size())]
          dist.all_gather(all_visual_embs, visual_embs)
          dist.all_gather(all_last_embedding, last_embedding)
          # Overwrite with embeddings produced on this replace, which have the gradient.
          all_visual_embs[dist.get_rank()] = visual_embs
          all_last_embedding[dist.get_rank()] = last_embedding
          visual_embs = torch.cat(all_visual_embs)
          last_embedding = torch.cat(all_last_embedding)

          start_idx = args.rank * images.shape[0]
          end_idx = start_idx + images.shape[0]

        print(visual_embs.shape, last_embedding.shape)
        logits_per_image = visual_embs @ last_embedding.t()
        logits_per_text = logits_per_image.t()
        if i == 0:
          print(f'Running contrastive loss over logits_per_text.shape = {logits_per_text.shape} and logits_per_image.shape = {logits_per_image.shape}')

        caption_loss = losses_utils.contrastive_loss(logits_per_text)
        image_loss = losses_utils.contrastive_loss(logits_per_image)
        caption_acc1, caption_acc5 = losses_utils.contrastive_acc(logits_per_text, topk=(1, 5))
        image_acc1, image_acc5 = losses_utils.contrastive_acc(logits_per_image, topk=(1, 5))
        loss += args.ret_loss_scale * (caption_loss + image_loss) / 2.0
        cont_losses.update(loss.item(), images.size(0))

        # measure accuracy and record loss
        top1_caption.update(caption_acc1[0], images.size(0))
        top5_caption.update(caption_acc5[0], images.size(0))
        top1_image.update(image_acc1[0], images.size(0))
        top5_image.update(image_acc5[0], images.size(0))
      elif model_mode == 'generation':
        if args.num_tokens != 0 and args.num_clip_tokens != args.num_tokens:
          seq_len = clip_emb.shape[1]
          last_embedding = last_embedding.reshape((last_embedding.shape[0], seq_len, -1))
          assert last_embedding.shape == clip_emb.shape, (last_embedding.shape == clip_emb.shape)

        image_loss = losses_utils.l2_loss(clip_emb, last_embedding)  # (N,)
        gen_loss = args.gen_loss_scale * image_loss.mean()
        loss += gen_loss
        gen_losses.update(gen_loss.item(), images.size(0))


      if model_mode == 'retrieval':
        ret_vis_emb_norm.update(visual_embs_norm.item(), images.size(0))
      elif model_mode == 'captioning':
        cap_vis_emb_norm.update(visual_embs_norm.item(), images.size(0))

      inp_emb_norm.update(input_embs_norm.item(), images.size(0))

      if model_mode in ['retrieval', 'generation']:
        ret_time.update(time.time() - mode_start)
      elif model_mode == 'captioning':
        cap_time.update(time.time() - mode_start)

    loss = loss / args.grad_accumulation_steps
    losses.update(loss.item(), images.size(0))
    loss.backward()

    # Update weights
    if ((i + 1) % args.grad_accumulation_steps == 0) or (i == args.steps_per_epoch - 1):
      # Zero out gradients of the embedding matrix outside of [GEN].
      for param in model.module.model.input_embeddings.parameters():
        assert param.grad.shape[0] == len(tokenizer)
        # Keep other embeddings frozen.
        mask = torch.zeros((param.grad.shape[0], 1)).to(param.grad)
        for ret_idx in args.retrieval_token_idx:
          mask[ret_idx] = 1
        for gen_idx in args.gen_token_idx:
          mask[gen_idx] = 1
        param.grad = param.grad * mask

      # compute gradient and do SGD step
      if args.grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      optimizer.step()
      optimizer.zero_grad()
      print('=' * 80)

    with torch.no_grad():
      # Normalize trainable embeddings.
      frozen_norm = torch.norm(model.module.model.input_embeddings.weight[:-args.num_tokens, :], dim=1).mean(0)
      for ret_idx in args.retrieval_token_idx:
        trainable_weight = model.module.model.input_embeddings.weight[ret_idx, :]
        model.module.model.input_embeddings.weight[ret_idx, :].div_(trainable_weight.norm(dim=-1) / frozen_norm)

      # Log norms to Tensorboard.
      embedding_norm = torch.norm(model.module.model.input_embeddings.weight, dim=1).mean()
      ret_embedding_norm = torch.norm(model.module.model.input_embeddings.weight[args.retrieval_token_idx, :], dim=-1).mean()
      all_emb_norm.update(embedding_norm.item(), images.size(0))
      ret_emb_norm.update(ret_embedding_norm.item(), images.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if actual_step == 1 or (i + 1) % args.print_freq == 0:
      print('First 5 values of first 3 tokens of embedding matrix:', model.module.model.input_embeddings.weight.data[:3, :5])
      if args.num_tokens > 0:
        print('First 5 values of [GEN0] token embeddings:', model.module.model.input_embeddings.weight.data[args.gen_token_idx[0], :5])
        print(f'First 5 values of [GEN{args.num_tokens-1}] token embeddings:', model.module.model.input_embeddings.weight.data[args.gen_token_idx[-1], :5])
      print('First 5 values of first [IMG0] token embeddings:', model.module.model.input_embeddings.weight.data[args.retrieval_token_idx[0], :5])
      print(f'First 5 values of first [IMG{args.num_tokens-1}] token embeddings:', model.module.model.input_embeddings.weight.data[args.retrieval_token_idx[-1], :5])

      ex_per_sec = args.batch_size / batch_time.avg
      if args.distributed:
        batch_time.all_reduce()
        data_time.all_reduce()
        ex_per_sec = (args.batch_size / batch_time.avg) * ngpus_per_node

        losses.all_reduce()
        ce_losses.all_reduce()
        top1.all_reduce()
        top5.all_reduce()
        cap_vis_emb_norm.all_reduce()
        ret_vis_emb_norm.all_reduce()
        inp_emb_norm.all_reduce()
        ret_time.all_reduce()
        all_emb_norm.all_reduce()
        ret_emb_norm.all_reduce()
        cont_losses.all_reduce()
        gen_losses.all_reduce()
        top1_caption.all_reduce()
        top5_caption.all_reduce()
        top1_image.all_reduce()
        top5_image.all_reduce()
        cap_time.all_reduce()

      progress.display(i + 1)

      writer.add_scalar('train/loss', losses.avg, actual_step)
      writer.add_scalar('train/ce_loss', ce_losses.avg, actual_step)
      writer.add_scalar('train/seq_top1_acc', top1.avg, actual_step)
      writer.add_scalar('train/seq_top5_acc', top5.avg, actual_step)
      writer.add_scalar('train/gen_l2_loss', gen_losses.avg, actual_step)
      writer.add_scalar('train/contrastive_loss', cont_losses.avg, actual_step)

      writer.add_scalar('train/t2i_top1_acc', top1_caption.avg, actual_step)
      writer.add_scalar('train/t2i_top5_acc', top5_caption.avg, actual_step)
      writer.add_scalar('train/i2t_top1_acc', top1_image.avg, actual_step)
      writer.add_scalar('train/i2t_top5_acc', top5_image.avg, actual_step)
      writer.add_scalar('train/embmat_all_norm', embedding_norm.item(), actual_step)
      writer.add_scalar('train/embmat_ret_norm', ret_embedding_norm.item(), actual_step)
      writer.add_scalar('train/vis_emb_norm_cap', cap_vis_emb_norm.avg, actual_step)
      writer.add_scalar('train/vis_emb_norm_ret', ret_vis_emb_norm.avg, actual_step)
      writer.add_scalar('train/text_emb_norm', inp_emb_norm.avg, actual_step)
      writer.add_scalar('metrics/total_secs_per_batch', batch_time.avg, actual_step)
      writer.add_scalar('metrics/total_secs_captioning', cap_time.avg, actual_step)
      writer.add_scalar('metrics/total_secs_retrieval', ret_time.avg, actual_step)
      writer.add_scalar('metrics/data_secs_per_batch', data_time.avg, actual_step)
      writer.add_scalar('metrics/examples_per_sec', ex_per_sec, actual_step)

      if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        and args.rank % ngpus_per_node == 0):
        image_bs = images.shape[0]
        normalized_images = images - images.min()
        normalized_images /= normalized_images.max()  # (N, 3, H, W)
        max_images_to_show = 16

        # Append caption text.
        pred_tokens = output[:, args.n_visual_tokens-1:-1, :].argmax(dim=-1)
        generated_captions = tokenizer.batch_decode(pred_tokens, skip_special_tokens=False)

        if model_mode == 'captioning':
          # OPTIM(jykoh): Truncate before creating images, rather than after. Some compute might be saved.
          # Create generated caption text.
          generated_cap_images = torch.stack([
            utils.create_image_of_text(
              generated_captions[i].encode('ascii', 'ignore'),
              width=normalized_images.shape[3],
              color=(255, 255, 0))
            for i in range(len(generated_captions))], axis=0)

          # Duplicate captions if we concatenated them.
          if (args.concat_captions_prob > 0 and model_mode == 'captioning' and generated_cap_images.shape[0] != caption_images.shape[0]):
            generated_cap_images = torch.cat([generated_cap_images, generated_cap_images], axis=0)

          display_images = torch.cat([normalized_images.float().cpu(), caption_images, generated_cap_images], axis=2)[:max_images_to_show]
          grid = torchvision.utils.make_grid(display_images, nrow=int(max_images_to_show ** 0.5), padding=4)
          writer.add_image('train/images_gen_cap', grid, actual_step)

        # Retrieved images (from text).
        retrieved_image_idx = logits_per_text[:image_bs, :image_bs].argmax(-1)
        t2i_images = torch.stack(
          [normalized_images[retrieved_image_idx[i], ...] for i in range(len(retrieved_image_idx))],
          axis=0)
        t2i_images = torch.cat([t2i_images.float().cpu(), caption_images], axis=2)[:max_images_to_show]
        t2i_grid = torchvision.utils.make_grid(t2i_images, nrow=int(max_images_to_show ** 0.5), padding=4)
        writer.add_image('train/t2i_ret', t2i_grid, actual_step)

        # Retrieved text (from image).
        retrieved_text_idx = logits_per_image[:image_bs, :image_bs].argmax(-1)
        retrieved_text = torch.stack(
          [caption_images[retrieved_image_idx[i], ...] for i in range(len(retrieved_text_idx))],
          axis=0)
        i2t_images = torch.cat([normalized_images.float().cpu(), retrieved_text], axis=2)[:max_images_to_show]
        i2t_grid = torchvision.utils.make_grid(i2t_images, nrow=int(max_images_to_show ** 0.5), padding=4)
        writer.add_image('train/i2t_ret', i2t_grid, actual_step)

      batch_time.reset()
      cap_time.reset()
      ret_time.reset()
      data_time.reset()
      losses.reset()
      ce_losses.reset()
      top1.reset()
      top5.reset()
      ret_vis_emb_norm.reset()
      cap_vis_emb_norm.reset()
      inp_emb_norm.reset()
      all_emb_norm.reset()
      ret_emb_norm.reset()
      cont_losses.reset()
      gen_losses.reset()
      top1_caption.reset()
      top5_caption.reset()
      top1_image.reset()
      top5_image.reset()

    if i == args.steps_per_epoch - 1:
      break

    scheduler.step()
    curr_lr = scheduler.get_last_lr()
    if (actual_step == 1) or (i + 1) % args.print_freq == 0:
      # Write current learning rate to Tensorboard.
      writer = SummaryWriter(args.log_dir)
      writer.add_scalar('train/lr', curr_lr[0], actual_step)
      writer.close()

  writer.close()


# Disable tokenizer parallelism.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == '__main__':
  main(sys.argv[1:])