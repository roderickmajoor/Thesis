import json

# Read the existing JSON file
with open('/media/roderickmajoor/TREKSTOR/Train/annotations.json', 'r') as f:
    coco_format = json.load(f)

# Remove the 'partX/' prefix from the 'file_name' field for each image
for image in coco_format['images']:
    image['file_name'] = image['file_name'].split('/')[1]  # Split and take the second part

# Save the updated JSON file
with open('/media/roderickmajoor/TREKSTOR/Train/annotations_updated.json', 'w') as f:
    json.dump(coco_format, f, indent=4)
