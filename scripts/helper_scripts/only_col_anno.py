import json

def filter_annotations(input_json, output_json, target_category_id):
    with open(input_json, 'r') as f:
        data = json.load(f)

    filtered_annotations = [ann for ann in data['annotations'] if ann['category_id'] == target_category_id]

    data['annotations'] = filtered_annotations
    data['categories'] = [cat for cat in data['categories'] if cat['id'] == target_category_id]

    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)

input_json = '/media/roderickmajoor/TREKSTOR/Test/annotations_pred.json'
output_json = '/media/roderickmajoor/TREKSTOR/Test/annotations_pred.json'
target_category_id = 0  # Replace with your target category ID

filter_annotations(input_json, output_json, target_category_id)
