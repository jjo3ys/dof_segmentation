import json

with open('data/test/annotations/instances_default.json', 'r') as f:
    json_data = json.load(f)
    annotations =json_data['annotations']
    images = json_data['images']

    origin_id = {}
    for i, image in enumerate(images):
        origin_id[image['id']]=i
        json_data['images'][i]['id'] = i
    
    for i, annotation in enumerate(annotations):
        json_data['annotations'][i]['image_id'] = origin_id[annotation['image_id']]
with open('data/test/annotations/instances_ordered.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f)