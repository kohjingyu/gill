"""Uses GILL to generate images for VisDial dialogue sequences.

Example usage:
  python generate_visdial_images.py  gill_visdial_outputs
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


# Download the VisDial validation annotations (https://www.dropbox.com/s/ibs3a0zhw74zisc/visdial_1.0_val.zip?dl=0),
# the dense answer annotations (https://www.dropbox.com/s/3knyk09ko4xekmc/visdial_1.0_val_dense_annotations.json?dl=0)
# and the images (https://www.dropbox.com/s/twmtutniktom7tu/VisualDialog_val2018.zip?dl=0).
# Extract everything to the `VisualDialog` folder.
visdial_dir = 'VisualDialog/'


if __name__ == '__main__':
    output_dir = sys.argv[1]
    os.makedirs(output_dir, exist_ok=True)
    print('Saving to', output_dir)

    model = models.load_gill('checkpoints/gill_opt/', load_ret_embs=False)
    g_cuda = torch.Generator(device='cuda').manual_seed(42)  # Fix the random seed.

    # Load VisDial data.
    split = 'val'
    img_dir = os.path.join(visdial_dir, f'VisualDialog_{split}2018/')
    with open(os.path.join(visdial_dir, f'visdial_1.0_{split}.json'), 'r') as f:
        visdial_data = json.load(f)
    with open(os.path.join(visdial_dir, f'visdial_1.0_{split}_dense_annotations.json'), 'r') as f:
        dense_data = json.load(f)
    
    # Sanity check for VisDial dialogues.
    assert len(dense_data) == len(visdial_data['data']['dialogs'])
    for i in range(len(dense_data)):
        assert dense_data[i]['image_id'] == visdial_data['data']['dialogs'][i]['image_id']

    questions = visdial_data['data']['questions']
    answers = visdial_data['data']['answers']
    dialogs = visdial_data['data']['dialogs']

    for example_idx in tqdm(range(len(dialogs))):
        dialog = dialogs[example_idx]
        image_id = str(dialog['image_id']).rjust(12, '0')
        contexts = []

        for i in range(len(dialog['dialog'])):
            contexts.append('Q: ' + questions[dialog['dialog'][i]['question']] + '?')
            contexts.append('A: ' + answers[dialog['dialog'][i]['answer']])

        cond_caption = '\n'.join(contexts)

        # Print inputs for the first few examples.
        if example_idx < 3:
            print(cond_caption)

        return_outputs = model.generate_for_images_and_texts(
            [cond_caption], num_words=2, gen_scale_factor=1e5, generator=g_cuda)
        gen_img = return_outputs[1]['gen'][0][0]
        with open(os.path.join(output_dir, f'{image_id}.png'), 'wb') as wf:
            gen_img.save(wf)

