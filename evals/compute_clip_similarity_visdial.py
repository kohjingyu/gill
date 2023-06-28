import json
import os

import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


# Define the paths to the groundtruth / generated image directories.
gen_img_dir = 'gill_visdial_outputs/'
visdial_dir = 'VisualDialog/'

if __name__ == "__main__":
    # Load CLIP model.
    device = 'cuda'
    model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(model_name)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model.to(device)

    # Load VisDial data.
    split = 'val'
    gt_img_dir = os.path.join(visdial_dir, f'VisualDialog_{split}2018/')
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

    all_scores = []

    for example_idx in tqdm(range(len(dialogs))):
        dialog = dialogs[example_idx]
        gt_image_id = str(dialog['image_id']).rjust(12, '0')
        gt_img_path = os.path.join(gt_img_dir, f'VisualDialog_{split}2018_{gt_image_id}.jpg')
        gen_img_path = os.path.join(gen_img_dir, f'{gt_image_id}.png')

        if not os.path.exists(gt_img_path) or not os.path.exists(gen_img_path):
            print(f'Skipping example {example_idx} because one of {gt_img_path} or {gen_img_path} does not exist.')
        else:
            # Load groundtruth image and compute its CLIP image features
            with open(gt_img_path, 'rb') as f:
                img = Image.open(f)
                inputs = clip_processor(images=img, return_tensors="pt")
                inputs = {k: v.cuda() for k, v in inputs.items()}
                gt_feat = clip_model.get_image_features(**inputs)

            # Compute generated image features.
            with open(gen_img_path, 'rb') as f:
                img = Image.open(f)
                inputs = clip_processor(images=img, return_tensors="pt")
                inputs = {k: v.cuda() for k, v in inputs.items()}
                image_feat = clip_model.get_image_features(**inputs)

            # Compute cosine similarity.
            score = ((image_feat / image_feat.norm()) @ (gt_feat / gt_feat.norm()).T).item()
            all_scores.append(score)

    score = np.mean(all_scores)
    print('CLIP similarity:', score)

    with open('visdial_clip_similarity.txt', 'w') as wf:
        wf.write(str(score))

