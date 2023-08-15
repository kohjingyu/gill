"""Runs Stable Diffusion v1.5 to generate images for PartiPrompts.

Example usage:
python scripts/generate_sd_p2_images.py    data/PartiPromptsAllDecisions.tsv    partiprompts_sd_v1.5_outputs
"""

import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline

p2_fn = sys.argv[1]
output_dir = sys.argv[2]
batch_size = 16


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    # NOTE: You may have to uncomment this line to override the safety checker
    # to avoid getting empty images for certain prompts.
    # pipe.safety_checker = lambda images, clip_input: (images, False)

    # Load PartiPrompts.
    with open(p2_fn, 'r') as f:
        captions = []
        filenames = []

        for i, line in enumerate(f.readlines()[1:]):
            data = line.strip().split('\t')
            captions.append(data[0])
            filenames.append(f'{i}.png')
    
    g_cuda = torch.Generator(device='cuda').manual_seed(1337)

    # Generate images in batches.
    num_batches = int(np.ceil(len(filenames) / batch_size))
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        images = pipe(captions[start_idx:end_idx], generator=g_cuda).images
        
        for j, fn in enumerate(filenames[start_idx:end_idx]):
            with open(os.path.join(output_dir, fn), 'wb') as wf:
                images[j].save(wf)