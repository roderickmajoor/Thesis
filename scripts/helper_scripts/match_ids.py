import json

# Load the ground truth annotations
with open('/media/roderickmajoor/TREKSTOR/Test/annotations_only_extended_cols_mask.json') as f:
    coco_gt = json.load(f)

# Load the predicted annotations
with open('/media/roderickmajoor/TREKSTOR/Test/annotations_pred.json') as f:
    coco_pred = json.load(f)

#Create a mapping from file_name to image_id in the ground truth
file_name_to_id_gt = {image['file_name']: image['id'] for image in coco_gt['images']}

# Create a mapping of old_id to new_id for the images in predicted annotations
old_to_new_id = {}
for image in coco_pred['images']:
    if image['file_name'] in file_name_to_id_gt:
        old_to_new_id[image['id']] = file_name_to_id_gt[image['file_name']]
        image['id'] = file_name_to_id_gt[image['file_name']]

# Update image ids in the 'annotations' section of predicted annotations
for ann in coco_pred['annotations']:
    if ann['image_id'] in old_to_new_id:
        ann['image_id'] = old_to_new_id[ann['image_id']]

# Save the updated predicted annotations to a new file
with open('/media/roderickmajoor/TREKSTOR/Test/annotations_pred_updated.json', 'w') as f:
    json.dump(coco_pred, f)

print("Updated image IDs in predicted annotations.")
