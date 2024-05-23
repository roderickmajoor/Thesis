import os
import xml.etree.ElementTree as ET
import json
import cv2

def extract_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = []

    # Extract TableRegion annotations
    for table_region in root.findall('.//{*}TableRegion'):
        region_id = table_region.attrib['id']
        region_coords = table_region.find('{*}Coords').attrib['points']

        annotation = {
            "id": region_id,
            "image_id": None,  # Will be populated later
            "category_id": 0,  # Adjust category_id as needed
            "bbox": [],  # Adjust bbox as needed
            "area": None,  # Will be populated later
            "segmentation": [],  # Adjust segmentation as needed
            "iscrowd": 0
        }

        # Parse coordinates
        points = region_coords.split()
        xs = []
        ys = []
        for point in points:
            x, y = map(int, point.split(','))
            xs.append(x)
            ys.append(y)

        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)

        annotation["bbox"] = [x_min, y_min, x_max - x_min, y_max - y_min]
        annotation["area"] = (x_max - x_min) * (y_max - y_min)

        annotations.append(annotation)

    # Extract TableCell annotations
    for table_cell in root.findall('.//{*}TableCell'):
        cell_id = table_cell.attrib['id']
        cell_coords = table_cell.find('{*}Coords').attrib['points']

        annotation = {
            "id": cell_id,
            "image_id": None,  # Will be populated later
            "category_id": 1,  # Adjust category_id as needed
            "bbox": [],  # Adjust bbox as needed
            "area": None,  # Will be populated later
            "segmentation": [],  # Adjust segmentation as needed
            "iscrowd": 0
        }

        # Parse coordinates
        points = cell_coords.split()
        xs = []
        ys = []
        for point in points:
            x, y = map(int, point.split(','))
            xs.append(x)
            ys.append(y)

        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)

        annotation["bbox"] = [x_min, y_min, x_max - x_min, y_max - y_min]
        annotation["area"] = (x_max - x_min) * (y_max - y_min)

        annotations.append(annotation)

    # Extract TextRegion annotations
    for text_region in root.findall('.//{*}TextRegion'):
        region_id = text_region.attrib['id']
        region_coords = text_region.find('{*}Coords').attrib['points']

        annotation = {
            "id": region_id,
            "image_id": None,  # Will be populated later
            "category_id": 2,  # Adjust category_id as needed
            "bbox": [],  # Adjust bbox as needed
            "area": None,  # Will be populated later
            "segmentation": [],  # Adjust segmentation as needed
            "iscrowd": 0
        }

        # Parse coordinates
        points = region_coords.split()
        xs = []
        ys = []
        for point in points:
            x, y = map(int, point.split(','))
            xs.append(x)
            ys.append(y)

        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)

        annotation["bbox"] = [x_min, y_min, x_max - x_min, y_max - y_min]
        annotation["area"] = (x_max - x_min) * (y_max - y_min)

        annotations.append(annotation)

    return annotations

def get_image_dimensions(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    return height, width

def generate_json(source_dir):
    images = []
    annotations = []

    image_id_counter = 0
    annotation_id_counter = 0

    # Loop through image files
    for image_file in os.listdir(source_dir):
        if image_file.endswith('.jpg'):
            image_name = image_file.split('.')[0]
            image_path = os.path.join(source_dir, image_file)
            xml_file = os.path.join(source_dir, 'page', f'{image_name}.xml')

            if os.path.exists(xml_file):
                # Add image information
                image_info = {
                    "id": image_id_counter,
                    "license": 1,
                    "file_name": image_file,
                    "height": None,  # Provide image height
                    "width": None,  # Provide image width
                    "date_captured": "2020-01-01T00:00:00+00:00"
                }
                images.append(image_info)

                # Extract annotations
                image_annotations = extract_annotations(xml_file)
                for annotation in image_annotations:
                    # Populate image_id field in annotation
                    annotation["image_id"] = image_id_counter
                    # Populate height and width in image_info if not already populated
                    if image_info["height"] is None or image_info["width"] is None:
                        image_info["height"], image_info["width"] = get_image_dimensions(image_path)
                    # Assign unique annotation id
                    annotation["id"] = annotation_id_counter
                    annotations.append(annotation)
                    annotation_id_counter += 1

                image_id_counter += 1

    # Create COCO format dictionary
    coco_format = {
        "info": {
            "year": "2020",
            "version": "1",
            "description": "Exported from roboflow.ai",
            "contributor": "Roboflow",
            "url": "https://app.roboflow.ai/datasets/hard-hat-sample/1",
            "date_created": "2000-01-01T00:00:00+00:00"
        },
        "licenses": [
            {
                "id": 1,
                "url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "name": "Public Domain"
            }
        ],
        "categories": [
            {"id": 0, "name": "TableRegion", "supercategory": "none"},
            {"id": 1, "name": "TableCell", "supercategory": "none"},
            {"id": 2, "name": "TextRegion", "supercategory": "none"}
        ],
        "images": images,
        "annotations": annotations
    }

    with open('/home/roderickmajoor/Desktop/Master/Thesis/Train_Test/Test/annotations.json', 'w') as json_file:
        json.dump(coco_format, json_file, indent=4)

# Path to the directory containing train/test images and XML files
source_dir = "/home/roderickmajoor/Desktop/Master/Thesis/Train_Test/Test"

# Generate JSON annotations in COCO format
generate_json(source_dir)
