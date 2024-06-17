import os
import xml.etree.ElementTree as ET
import json
import cv2
import numpy as np
from scipy.spatial import ConvexHull
from vertical_line_detector import foreground_extractor

def extract_annotations(xml_file, image_path, start_annotation_id):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    height, width = get_image_dimensions(image_path)

    image = cv2.imread(image_path)
    fg_image, foreground = foreground_extractor(image.copy())
    fg_y = foreground[1]
    fg_h = foreground[3]

    """
    # Initialize list to store all annotations
    annotations = []

    # Iterate through each TableRegion
    for table_region in root.findall('.//{*}TableRegion'):
        region_id = table_region.attrib['id']

        # Initialize dictionaries to store row and column data for the current TableRegion
        rows = {}
        columns = {}

        # Extract TableCell annotations within the current TableRegion
        for table_cell in table_region.findall('.//{*}TableCell'):
            cell_id = table_cell.attrib['id']
            cell_coords = table_cell.find('{*}Coords').attrib['points']
            row = int(table_cell.attrib['row'])
            col = int(table_cell.attrib['col'])

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

            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            # Add to rows dictionary
            if row not in rows:
                rows[row] = {'xs': [], 'ys': []}
            rows[row]['xs'].extend(xs)
            rows[row]['ys'].extend(ys)

            # Add to columns dictionary
            if col not in columns:
                columns[col] = {'xs': [], 'ys': []}
            columns[col]['xs'].extend(xs)
            columns[col]['ys'].extend(ys)
    """
    # Initialize list to store all annotations
    annotations = []

    # List to store TableRegions information
    table_regions_info = []
    id = start_annotation_id

    # Iterate through each TableRegion
    for table_region in root.findall('.//{*}TableRegion'):
        region_id = table_region.attrib['id']

        # Initialize dictionaries to store row and column data for the current TableRegion
        rows = {}
        columns = {}

        # Extract TableCell annotations within the current TableRegion
        for table_cell in table_region.findall('.//{*}TableCell'):
            cell_id = table_cell.attrib['id']
            cell_coords = table_cell.find('{*}Coords').attrib['points']
            row = int(table_cell.attrib['row'])
            col = int(table_cell.attrib['col'])

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

            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            # Add to rows dictionary
            if row not in rows:
                rows[row] = {'xs': [], 'ys': []}
            rows[row]['xs'].extend(xs)
            rows[row]['ys'].extend(ys)

            # Add to columns dictionary
            if col not in columns:
                columns[col] = {'xs': [], 'ys': [], 'x_min': float('inf'), 'x_max': float('-inf')}
            columns[col]['xs'].extend(xs)
            columns[col]['ys'].extend(ys)
            columns[col]['x_min'] = min(columns[col]['x_min'], x_min)
            columns[col]['x_max'] = max(columns[col]['x_max'], x_max)



        # Calculate bounding boxes for columns within the current TableRegion
        for i, (key, coords) in enumerate(columns.items()):
            anno_id = id + i
            print(anno_id)

            x_min = coords['x_min']
            x_max = coords['x_max']

            bbox = [x_min, fg_y, x_max - x_min, fg_h]
            area = (x_max - x_min) * height

            # Create segmentation using bounding box points extended to top and bottom of the page
            segmentation = [
                x_min, fg_y,
                x_max, fg_y,
                x_max, fg_h,
                x_min, fg_h
            ]

            annotation = {
                "id": anno_id,
                "image_id": None,  # Will be populated later
                "category_id": 0,  # Adjust category_id as needed
                "bbox": bbox,
                "area": area,
                "segmentation": [segmentation],  # Adjust segmentation as needed
                "iscrowd": 0
            }

            annotations.append(annotation)

        id += len(columns)

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
                image_annotations = extract_annotations(xml_file, image_path, annotation_id_counter)
                for annotation in image_annotations:
                    # Populate image_id field in annotation
                    annotation["image_id"] = image_id_counter
                    # Populate height and width in image_info if not already populated
                    if image_info["height"] is None or image_info["width"] is None:
                        image_info["height"], image_info["width"] = get_image_dimensions(image_path)
                    # Assign unique annotation id
                    #annotation["id"] = annotation_id_counter
                    annotations.append(annotation)

                image_id_counter += 1
                annotation_id_counter += len(image_annotations)

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
            {
                "id": 0,
                "name": "Column",
                "supercategory": "none"
            },
            {
                "id": 1,
                "name": "Row",
                "supercategory": "none"
            }
        ],
        "images": images,
        "annotations": annotations
    }

    with open('/media/roderickmajoor/TREKSTOR/Test/annotations_mask_cols.json', 'w') as json_file:
        json.dump(coco_format, json_file, indent=4)

# Path to the directory containing train/test images and XML files
source_dir = "/media/roderickmajoor/TREKSTOR/Test"

# Generate JSON annotations in COCO format
generate_json(source_dir)
