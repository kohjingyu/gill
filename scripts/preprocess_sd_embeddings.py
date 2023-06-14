"""This script extracts text embeddings from the text encoder of Stable Diffusion
for a given dataset of captions, and saves them to disk.
The outputs are used in training GILL.

Example usage:
python scripts/preprocess_sd_embeddings.py  datasets/cc3m_val.tsv data/cc3m/validation/clip_embs
"""

import numpy as np
import os
import sys
from joblib import Parallel, delayed
from tqdm import tqdm
import torch

# Load a slightly modified version of the Stable Diffusion pipeline.
# This allows us to extract text embeddings directly (without generating images).
from gill.custom_sd import StableDiffusionPipeline


# Default arguments for running preprocessing.
model_id = "runwayml/stable-diffusion-v1-5"
batch_size = 128
input_captions_fp = sys.argv[1]  # tab separated file of captions and image ids
clip_output_dir = sys.argv[2]  # output directory to save clip embeddings in
os.makedirs(clip_output_dir, exist_ok=True)


def save_to_path(emb, path):
    """Save embeddings to disk."""
    try:
        with open(path, 'wb') as wf:
            np.save(wf, emb)
    except:
        print("Error with", path)
    return path


if __name__ == '__main__':
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    if not torch.cuda.is_available():
        print('WARNING: using CPU, this will be slow!')
    else:
        pipe = pipe.to("cuda")

    # Get existing files, so that we don't recompute them.
    existing_files = set([f.strip('.npy') for f in os.listdir(clip_output_dir)])

    # Load captions and associated image ids.
    with open(input_captions_fp, 'r') as f:
        data = f.readlines()
        examples = data[1:]
        captions = []
        image_ids = []

        for x in examples:
            d = x.strip().split('\t')
            if d[1] not in existing_files:
                captions.append(d[0])
                image_ids.append(d[1])
        assert len(captions) == len(image_ids)

    # Extract embeddings in batches.
    num_batches = int(np.ceil(len(captions) / batch_size))
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_captions = captions[start_idx:end_idx]
        batch_ids = image_ids[start_idx:end_idx]
        prompt_embeds = pipe(batch_captions, return_prompts_only=True).detach().cpu().numpy()        

        # Save embeddings to disk in parallel.
        Parallel(n_jobs=8)(delayed(save_to_path)(
            prompt_embeds[j, :, ...], os.path.join(clip_output_dir, f'{batch_ids[j]}.npy')
        ) for j in range(prompt_embeds.shape[0]))
