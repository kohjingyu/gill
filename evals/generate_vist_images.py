"""Uses GILL to generate images for VIST interleaved image + text sequences.

Example usage:
  python generate_vist_images.py  gill_vist_outputs
"""

from collections import namedtuple
import json
import os
import pickle as pkl
import sys

from PIL import Image
import torch
from tqdm import tqdm

from gill import models


# Path containing the VIST images.
vist_image_dir = 'sis/val_images/'
# Path containing the formatted VIST annotations.
vist_data_path = 'sis/val_formatted.json'


if __name__ == '__main__':
    output_dir = sys.argv[1]
    os.makedirs(output_dir, exist_ok=True)
    print('Saving to', output_dir)

    model = models.load_gill('checkpoints/gill_opt/')
    g_cuda = torch.Generator(device='cuda').manual_seed(42)  # Fix the random seed.

    # Load VIST data.
    with open(vist_data_path, 'r') as f:
        vist_data = json.load(f)
        story_ids = list(vist_data['annotations'].keys())

    for story_idx, (story_id, story_data) in tqdm(enumerate(vist_data['annotations'].items()), total=len(vist_data['annotations'])):
        # Load all images except the last (we're generating the last one)
        image_paths = [os.path.join(vist_image_dir, s['image_id'] + '.png') for s in story_data][:-1]
        gt_image_id = story_data[-1]['image_id']
        captions = [s['caption'] for s in story_data]
        assert (len(image_paths) == len(captions) - 1) or (len(image_paths) == len(captions))

        should_process = True
        for path in image_paths:
            if not os.path.exists(path):
                print(f'Image not found: {path}. Skipping story {story_id}')
                should_process = False
                break
        
        if should_process:
            caption_range = range(len(captions))
            input_data = []

            for i_i, i in enumerate(caption_range):
                caption = captions[i]
                input_data.append(caption)

                if i < len(captions) - 1:  # Use first n-1 images
                    with open(image_paths[i], 'rb') as f:
                        img = Image.open(f).convert('RGB').resize((224, 224))
                        input_data.append(img)

            # Print outputs for first 3 examples as a sanity check.
            if story_idx < 3:
                print(input_data)

            # Set a really high ret scale so that we force the model to generate an image
            # This is equivalent to explicitly appending the [IMG] tokens to the input.
            return_outputs = model.generate_for_images_and_texts(
                input_data, num_words=2, gen_scale_factor=1e5, generator=g_cuda)
            
            # Save the generated image.
            generated_img = return_outputs[1]['gen'][0][0]
            with open(os.path.join(output_dir, f'{gt_image_id}.png'), 'wb') as f:
                generated_img.save(f)
                print("Saving to", os.path.join(output_dir, f'{gt_image_id}.png'))

