from get_cell_coords import get_cells
from shapely.geometry import Polygon
import json
import os
import cv2

def polygon_to_coco_annotation(polygon, image_id, category_id, annotation_id):
    segmentation = [list(sum(polygon.exterior.coords[:-1], ()))]
    min_x, min_y, max_x, max_y = polygon.bounds
    bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
    area = polygon.area
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": area,
        "segmentation": segmentation,
        "iscrowd": 0
    }
    return annotation

def one(loghi_xml, jpg_file, image_id, start_annotation_id):
    predicted_cells = get_cells(loghi_xml, jpg_file)
    annotations = []
    for i, polygon in enumerate(predicted_cells):
        annotation = polygon_to_coco_annotation(polygon, image_id, 1, start_annotation_id + i)
        annotations.append(annotation)
    return annotations

def all_images():
    dir2 = '/media/roderickmajoor/TREKSTOR/Train/images_parts'
    dir3 = '/media/roderickmajoor/TREKSTOR/Train/images_parts'
    subdirs = [d for d in os.listdir(dir2) if os.path.isdir(os.path.join(dir2, d))]

    images = []
    annotations = []
    image_id = 0
    annotation_id = 0

    for subdir in subdirs:
        print(subdir)
        files2 = os.listdir(os.path.join(dir2, subdir))
        files3 = os.listdir(os.path.join(dir3, subdir, 'page'))
        filenames = set(os.path.splitext(file)[0] for file in files2) & \
                    set(os.path.splitext(file)[0] for file in files3)

        for filename in filenames:
            jpg_file = os.path.join(dir2, subdir, filename + '.jpg')
            loghi_xml = os.path.join(dir3, subdir, 'page', filename + '.xml')

            # Get image dimensions using OpenCV
            image = cv2.imread(jpg_file)
            height, width, _ = image.shape

            # Add image entry
            image_entry = {
                "id": image_id,
                "license": 1,
                "file_name": os.path.relpath(jpg_file, dir2),
                "height": height,
                "width": width,
                "date_captured": "2020-01-01T00:00:00+00:00"
            }
            images.append(image_entry)

            # Get annotations for this image
            image_annotations = one(loghi_xml, jpg_file, image_id, annotation_id)
            annotations.extend(image_annotations)

            # Increment IDs
            image_id += 1
            annotation_id += len(image_annotations)

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
                "name": "TableRegion",
                "supercategory": "none"
            },
            {
                "id": 1,
                "name": "TableCell",
                "supercategory": "none"
            },
            {
                "id": 2,
                "name": "TextRegion",
                "supercategory": "none"
            }
        ],
        "images": images,
        "annotations": annotations
    }

    return coco_format

coco_format = all_images()

with open('/media/roderickmajoor/TREKSTOR/Train/annotations.json', 'w') as f:
    json.dump(coco_format, f, indent=4)
