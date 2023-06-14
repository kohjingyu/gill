"""Modified from https://github.com/mlfoundations/open_clip"""

from typing import Optional, Tuple, List

import collections
import logging
import os
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
from torchvision import transforms as T
from PIL import Image, ImageFont
from torch.utils.data import Dataset

from gill import utils


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_dataset(args, split: str, tokenizer, precision: str = 'fp32') -> Dataset:
  assert split in ['train', 'val'
    ], 'Expected split to be one of "train" or "val", got {split} instead.'

  dataset_paths = []
  image_data_dirs = []
  train = split == 'train'

  # Default configs for datasets.
  # Folder structure should look like:
  if split == 'train':
    if 'cc3m' in args.dataset:
      dataset_paths.append(os.path.join(args.dataset_dir, 'cc3m_train.tsv'))
      image_data_dirs.append(os.path.join(args.image_dir, 'cc3m/training/'))
    else:
      raise NotImplementedError

  elif split == 'val':
    if 'cc3m' in args.val_dataset:
      dataset_paths.append(os.path.join(args.dataset_dir, 'cc3m_val.tsv'))
      image_data_dirs.append(os.path.join(args.image_dir, 'cc3m/validation'))
    else:
      raise NotImplementedError

    assert len(dataset_paths) == len(image_data_dirs) == 1, (dataset_paths, image_data_dirs)
  else:
    raise NotImplementedError

  if len(dataset_paths) > 1:
    print(f'{len(dataset_paths)} datasets requested: {dataset_paths}')
    dataset = torch.utils.data.ConcatDataset([
      CsvDataset(path, image_dir, tokenizer, 'image',
        'caption', args.visual_model, train=train, max_len=args.max_len, precision=args.precision,
        image_size=args.image_size, retrieval_token_idx=args.retrieval_token_idx, gen_token_idx=args.gen_token_idx, 
        num_tokens=args.num_tokens, num_clip_tokens=args.num_clip_tokens)
      for (path, image_dir) in zip(dataset_paths, image_data_dirs)])
  elif len(dataset_paths) == 1:
    dataset = CsvDataset(dataset_paths[0], image_data_dirs[0], tokenizer, 'image',
      'caption', args.visual_model, train=train, max_len=args.max_len, precision=args.precision,
      image_size=args.image_size, retrieval_token_idx=args.retrieval_token_idx, gen_token_idx=args.gen_token_idx, 
      num_tokens=args.num_tokens, num_clip_tokens=args.num_clip_tokens)
  else:
    raise ValueError(f'There should be at least one valid dataset, got train={args.dataset}, val={args.val_dataset} instead.')
  return dataset


class CsvDataset(Dataset):
  def __init__(self, input_filename, base_image_dir, tokenizer, img_key,
               caption_key, feature_extractor_model: str,
               train: bool = True, max_len: int = 32, sep="\t", precision: str = 'fp32',
               image_size: int = 224, retrieval_token_idx: List[int] = [-1], gen_token_idx: List[int] = [-1],
               num_tokens: int = 1, num_clip_tokens: int = 1):
    logging.debug(f'Loading tsv data from {input_filename}.')
    df = pd.read_csv(input_filename, sep=sep)

    self.base_image_dir = base_image_dir
    self.images = df[img_key].tolist()
    self.captions = df[caption_key].tolist()
    assert len(self.images) == len(self.captions)

    self.feature_extractor_model = feature_extractor_model
    self.feature_extractor = utils.get_feature_extractor_for_model(
      feature_extractor_model, image_size=image_size, train=False)
    self.image_size = image_size

    self.tokenizer = tokenizer
    self.max_len = max_len
    self.precision = precision
    self.retrieval_token_idx = retrieval_token_idx
    self.gen_token_idx = gen_token_idx
    self.num_tokens = num_tokens
    self.num_clip_tokens = num_clip_tokens

    self.font = None

    logging.debug('Done loading data.')

  def __len__(self):
    return len(self.captions)

  def __getitem__(self, idx):
    while True:
      image_path = os.path.join(self.base_image_dir, str(self.images[idx]))
      caption = str(self.captions[idx])
      clip_l_path = os.path.join(self.base_image_dir, 'clip_embs', str(self.images[idx]) + '.npy')

      try:
        img = Image.open(image_path)
        images = utils.get_pixel_values_for_model(self.feature_extractor, img)

        # Only load if we are in generation mode.
        with open(clip_l_path, 'rb') as f:
          clip_emb = np.load(f, allow_pickle=True)   # (num_clip_tokens, 768)
          clip_emb = clip_emb[:self.num_clip_tokens, :]

        # Generation mode.
        caption = caption
        for i in range(self.num_tokens):
          caption += f'[IMG{i}]'
        tokenized_data = self.tokenizer(
          caption,
          return_tensors="pt",
          padding='max_length',
          truncation=True,
          max_length=self.max_len)
        tokens = tokenized_data.input_ids[0]
        caption_len = tokenized_data.attention_mask[0].sum()

        # If IMG tokens are overridden by padding, replace them with the correct token.
        if tokens[-1] not in [self.tokenizer.pad_token_id, self.gen_token_idx[-1]]:
          tokens[-self.num_tokens:] = torch.tensor(self.gen_token_idx).to(dtype=tokens.dtype, device=tokens.device)

        decode_caption = self.tokenizer.decode(tokens, skip_special_tokens=False)
        self.font = self.font or ImageFont.load_default()
        cap_img = utils.create_image_of_text(decode_caption.encode('ascii', 'ignore'), width=self.image_size, nrows=2, font=self.font)

        return image_path, images, cap_img, tokens, caption_len, tokens, caption_len, clip_emb
      except Exception as e:
        print(f'Error reading for {image_path} with caption {caption}: {e}')
        # Pick a new example at random.
        idx = np.random.randint(0, len(self)-1)
