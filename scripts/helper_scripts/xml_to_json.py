import os
import xml.etree.ElementTree as ET
import json

def extract_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = []

    # Extract TableRegion annotations
    for table_region in root.findall('.//{*}TableRegion'):
        region_id = table_region.attrib['id']
        region_coords = table_region.find('{*}Coords').attrib['points']

        annotation = {
            "shape_attributes": {"name": "polygon", "all_points_x": [], "all_points_y": []},
            "region_attributes": {"name": "TableRegion"}
        }

        # Parse coordinates
        points = region_coords.split()
        for point in points:
            x, y = map(int, point.split(','))
            annotation["shape_attributes"]["all_points_x"].append(x)
            annotation["shape_attributes"]["all_points_y"].append(y)

        annotations.append(annotation)

    # Extract TableCell annotations
    for table_cell in root.findall('.//{*}TableCell'):
        cell_id = table_cell.attrib['id']
        cell_coords = table_cell.find('{*}Coords').attrib['points']

        annotation = {
            "shape_attributes": {"name": "polygon", "all_points_x": [], "all_points_y": []},
            "region_attributes": {"name": "TableCell"}
        }

        # Parse coordinates
        points = cell_coords.split()
        for point in points:
            x, y = map(int, point.split(','))
            annotation["shape_attributes"]["all_points_x"].append(x)
            annotation["shape_attributes"]["all_points_y"].append(y)

        annotations.append(annotation)

    # Extract TextRegion annotations
    for text_region in root.findall('.//{*}TextRegion'):
        region_id = text_region.attrib['id']
        region_coords = text_region.find('{*}Coords').attrib['points']

        annotation = {
            "shape_attributes": {"name": "polygon", "all_points_x": [], "all_points_y": []},
            "region_attributes": {"name": "TextRegion"}
        }

        # Parse coordinates
        points = region_coords.split()
        for point in points:
            x, y = map(int, point.split(','))
            annotation["shape_attributes"]["all_points_x"].append(x)
            annotation["shape_attributes"]["all_points_y"].append(y)

        annotations.append(annotation)

    return annotations



def generate_json(source_dir):
    annotations_dict = {}

    # Loop through image files
    for image_file in os.listdir(source_dir):
        if image_file.endswith('.jpg'):
            image_name = image_file.split('.')[0]
            image_path = os.path.join(source_dir, image_file)
            xml_file = os.path.join(source_dir, 'page', f'{image_name}.xml')

            if os.path.exists(xml_file):
                annotations = extract_annotations(xml_file)
                annotations_dict[image_name] = {
                    "filename": image_file,
                    "size": os.path.getsize(image_path),
                    "regions": annotations,
                    "file_attributes": {"caption": "", "public_domain": "no", "image_url": ""}
                }

    with open('/home/roderickmajoor/Desktop/Master/Thesis/Train_Test/Test/annotations.json', 'w') as json_file:
        json.dump(annotations_dict, json_file, indent=4)

# Path to the directory containing train/test images and XML files
source_dir = "/home/roderickmajoor/Desktop/Master/Thesis/Train_Test/Test"

# Generate JSON annotations
generate_json(source_dir)
