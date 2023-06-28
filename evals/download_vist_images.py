import collections
import json
import os
import PIL
from tqdm import tqdm
from gill import utils

# Download the Visual Storytelling SIS dataset from https://visionandlanguage.net/VIST/json_files/story-in-sequence/SIS-with-labels.tar.gz
# Extract the files (there should be three sets: train, val, and test).
# We use the val set for reporting results.
vist_val_json_path = 'sis/val.story-in-sequence.json'
# Output directory to save images.
output_dir = 'sis/val_images/'
os.makedirs(output_dir, exist_ok=True)
# Path to save formatted annotations to.
val_formatted_path = 'sis/val_formatted.json'

if __name__ == '__main__':
    # Load VIST data.
    with open(vist_val_json_path, 'r') as f:
        vist_data_raw = json.load(f)
    # Format into a dictionary of {story_id: data} items.
    vist_data = {
        'annotations': collections.defaultdict(list)
    }
    used_image_ids = []

    for ann in vist_data_raw['annotations']:
        assert len(ann) == 1
        ann = ann[0]
        story_id = ann['story_id']
        vist_data['annotations'][story_id].append({
            'caption': ann['text'],
            'image_id': ann['photo_flickr_id'],
            'sequence_index': ann['worker_arranged_photo_order'],
        })
        used_image_ids.append(ann['photo_flickr_id'])

    used_image_ids = set(used_image_ids)
    # Save formatted annotations.
    with open(val_formatted_path, 'w') as wf:
        json.dump(vist_data, wf)

    # Map image ids to urls.
    id2url = {}
    for image_data in vist_data_raw['images']:
        image_id = image_data['id']
        if image_id in used_image_ids:
            image_url = image_data.get('url_o', None)
            if image_url is not None:
                id2url[image_id] = image_url

    # Download images.
    processed_images = set()
    print("Saving images to", output_dir)
    for story_idx, (story_id, story_data) in tqdm(enumerate(vist_data['annotations'].items()), total=len(vist_data['annotations'])):
        for s in story_data:
            image_id = s['image_id']
            if image_id not in processed_images:
                output_path = os.path.join(output_dir, f'{image_id}.png')

                # Save image if we have the url and it doesn't already exist.
                if image_id in id2url and not os.path.exists(output_path):
                    try:
                        image = utils.get_image_from_url(id2url[image_id])
                        # Save image to output dir.
                        with open(output_path, 'wb') as wf:
                            image.save(wf)
                    except PIL.UnidentifiedImageError:
                        print("Error saving image", image_id)

                processed_images.add(image_id)


