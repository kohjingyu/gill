import json
import os

import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


# Define the paths to the groundtruth / generated image directories.
gen_img_dir = 'gill_vist_outputs/'
gt_img_dir = "sis/val_images/"
vist_data_path = 'sis/val_formatted.json'

if __name__ == "__main__":
    # Load CLIP model.
    device = 'cuda'
    model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(model_name)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model.to(device)

    with open(vist_data_path, 'r') as f:
        vist_data = json.load(f)

    all_scores = []

    for story_idx, (story_id, story_data) in tqdm(enumerate(vist_data['annotations'].items()), total=len(vist_data['annotations'])):
        gt_image_id = story_data[-1]['image_id']
        gt_img_path = os.path.join(gt_img_dir, gt_image_id + '.png')
        gen_img_path = os.path.join(gen_img_dir, gt_image_id + '.png')

        if not os.path.exists(gt_img_path) or not os.path.exists(gen_img_path):
            print(f'Skipping example {story_id} because one of {gt_img_path} or {gen_img_path} does not exist.')
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

    with open('vist_clip_similarity.txt', 'w') as wf:
        wf.write(str(score))

