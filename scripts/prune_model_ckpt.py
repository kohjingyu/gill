"""Prunes model weights to keep just the necessary trained weights.

Example usage:
  python scripts/prune_model_ckpt.py  runs/gill_exp
"""

import json
import os
import sys
import torch


if __name__ == '__main__':
    model_dir = sys.argv[1]

    with open(os.path.join(model_dir, 'ckpt_best.pth.tar'), 'rb') as f:
        checkpoint = torch.load(f)

    with open(os.path.join(model_dir, 'model_args.json'), 'rb') as f:
        model_args = json.load(f)

    del checkpoint['epoch']
    del checkpoint['best_acc1']
    del checkpoint['optimizer']
    del checkpoint['scheduler']

    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        state_dict[k.replace('module.', '')] = v.detach().clone()

    checkpoint['state_dict'] = state_dict
    finetuned_tokens = checkpoint['state_dict']['model.input_embeddings.weight'][-model_args['num_gen_tokens']:, :].detach().clone()
    checkpoint['state_dict']['model.input_embeddings.weight'] = finetuned_tokens

    with open(os.path.join(model_dir, 'pretrained_ckpt.pth.tar'), 'wb') as f:
        torch.save(checkpoint, f)