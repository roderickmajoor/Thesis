# This script is used to create the pseudo-annotations in COCO format so that
# they can be used for a Detectron2 model.

from get_cell_coords import get_cells, get_col_row, get_col  # Import necessary functions for cell coordinates
from shapely.geometry import Polygon  # Import Polygon class for creating polygon objects
import json  # Import JSON module for handling JSON operations
import os  # Import OS module for file and directory operations
import cv2  # Import OpenCV for image processing
import numpy as np  # Import NumPy for numerical operations

def polygon_to_coco_annotation(polygon, image_id, category_id, annotation_id):
    """
    Converts a Shapely Polygon object into a COCO format annotation.

    Args:
    polygon (Polygon): The polygon to convert.
    image_id (int): The ID of the image to which this annotation belongs.
    category_id (int): The category ID of the object.
    annotation_id (int): The ID of the annotation.

    Returns:
    dict: A dictionary representing the annotation in COCO format.
    """
    segmentation = [list(sum(polygon.exterior.coords[:-1], ()))]  # Convert polygon exterior coordinates to segmentation format
    min_x, min_y, max_x, max_y = polygon.bounds  # Get bounding box coordinates
    bbox = [min_x, min_y, max_x - min_x, max_y - min_y]  # Create bounding box list
    area = polygon.area  # Calculate the area of the polygon
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
    """
    Processes a single image and its corresponding XML to extract column annotations.

    Args:
    loghi_xml (str): Path to the XML file containing column data.
    jpg_file (str): Path to the corresponding image file.
    image_id (int): The ID of the image.
    start_annotation_id (int): The starting ID for annotations.

    Returns:
    list: A list of annotations in COCO format for the given image.
    """
    column_coords = get_col(loghi_xml, jpg_file)  # Get column coordinates from XML and image

    # Convert the tuple of numpy arrays to a list of numpy arrays
    column_coords_list = list(column_coords)

    # Create a list to store the Polygon objects
    column_polygons = []

    # Convert each numpy array in the list to a Polygon object
    for coords in column_coords_list:
        # Reshape the array to (N, 2)
        coords_reshaped = np.reshape(coords, (-1, 2))

        # Convert the reshaped array to a list of tuples
        coords_tuples = [tuple(coord) for coord in coords_reshaped]

        # Create a Polygon object and add it to the list if it has 4 or more points
        if len(coords_tuples) >= 4:
            column_polygons.append(Polygon(coords_tuples))

    # Convert each polygon to a COCO annotation
    annotations = []
    for i, polygon in enumerate(column_polygons):
        annotation = polygon_to_coco_annotation(polygon, image_id, 0, start_annotation_id + i)
        annotations.append(annotation)

    return annotations

def all_images(directory):
    """
    Processes all images in specified directories to create COCO format annotations.

    Returns:
    dict: A dictionary in COCO format containing all image and annotation data.
    """
    dir = directory
    subdirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]  # Get all subdirectories

    images = []
    annotations = []
    image_id = 0
    annotation_id = 0

    for subdir in subdirs:
        print("Processing subdir: " + subdir)
        image_dir = os.listdir(os.path.join(dir, subdir))  # List files in directory
        xml_dir = os.listdir(os.path.join(dir, subdir, 'page'))  # List files in 'page' subdirectory
        filenames = set(os.path.splitext(file)[0] for file in image_dir) & \
                    set(os.path.splitext(file)[0] for file in xml_dir)  # Find common filenames

        for filename in filenames:
            jpg_file = os.path.join(dir, subdir, filename + '.jpg')
            loghi_xml = os.path.join(dir, subdir, 'page', filename + '.xml')

            # Get image dimensions using OpenCV
            image = cv2.imread(jpg_file)
            height, width, _ = image.shape

            # Add image entry
            image_entry = {
                "id": image_id,
                "license": 1,
                "file_name": os.path.relpath(jpg_file, dir).split('/')[1],  # Relative file path for image
                "height": height,
                "width": width,
                "date_captured": "2020-01-01T00:00:00+00:00"  # Placeholder date
            }
            images.append(image_entry)

            # Get annotations for this image
            image_annotations = one(loghi_xml, jpg_file, image_id, annotation_id)
            annotations.extend(image_annotations)

            # Increment IDs
            image_id += 1
            annotation_id += len(image_annotations)

    # Create final COCO format dictionary
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
            }
        ],
        "images": images,
        "annotations": annotations
    }

    return coco_format

# Fill in the directories of the image files and page folder.
# Folder is expected to be a folder containing subfolders.
# These subfolders are expected to have 1) .jpg files and 2) a folder called 'page' containing the corresponding loghi pageXML files.
directory = '/media/roderickmajoor/TREKSTOR/Train/images_parts'

# Generate COCO format annotations for all images
coco_format = all_images(directory)

# Save the COCO format annotations to a JSON file
with open('/media/roderickmajoor/TREKSTOR/temp1/Test/annotations_pred.json', 'w') as f:
    json.dump(coco_format, f, indent=4)
