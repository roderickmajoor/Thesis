import xml.etree.ElementTree as ET
import cv2
import numpy as np

ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

# Parse the XML file
gt_tree = ET.parse('/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/page/WBMA00007000010.xml')
image = cv2.imread('/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/WBMA00007000010.jpg')
gt_root = gt_tree.getroot()

# Dictionary to store merged areas for each column and table region
merged_areas_by_column_region = {}

# Iterate through TableRegion elements in the GT
for table_region in gt_root.findall('.//page:TableRegion', ns):
    # Dictionary to store merged areas for each column in the current table region
    merged_areas_by_column = {}

    # Iterate through TableCell elements
    for table_cell in table_region.findall('page:TableCell', ns):
        col_index = table_cell.attrib.get('col')
        cell_coords = [(int(point.split(',')[0]), int(point.split(',')[1])) for point in table_cell.find('page:Coords', ns).attrib.get('points').split()]

        # Update merged areas for the current column
        if col_index in merged_areas_by_column:
            merged_areas_by_column[col_index].append(cell_coords)
        else:
            merged_areas_by_column[col_index] = [cell_coords]

    # Store the merged areas for the current table region
    merged_areas_by_column_region[table_region.attrib.get('id')] = merged_areas_by_column

gt_bounding_boxes = []

# Draw lines connecting top-left, top-right, bottom-left, and bottom-right points for each column in each table region
for table_region_id, merged_areas_by_column in merged_areas_by_column_region.items():
    # Iterate through columns in the current table region
    for col_index, cell_coords_list in merged_areas_by_column.items():
        # Sort cells by their y-coordinate
        cell_coords_list.sort(key=lambda x: x[0][1])
        # Get the top-left, top-right, bottom-left, and bottom-right points for the column
        top_left = min(cell_coords_list[0], key=lambda x: x[0] + x[1])
        top_right = max(cell_coords_list[0], key=lambda x: x[0] - x[1])
        bottom_left = min(cell_coords_list[-1], key=lambda x: x[0] - x[1])
        bottom_right = max(cell_coords_list[-1], key=lambda x: x[0] + x[1])

        gt_bounding_boxes.append([top_left, top_right, bottom_right, bottom_left])

predicted_tree = ET.parse('/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/page/WBMA00007000010_columns_found.xml')
predicted_root = predicted_tree.getroot()

# Get the Page element
page_element = predicted_root.find('.//Page')

# Get all TextRegion elements under the Page
text_regions = page_element.findall('.//TextRegion')

predicted_bounding_boxes = []

for region in text_regions:
    coords = region.find('Coords').attrib['points']
    points = [(int(p.split(',')[0]), int(p.split(',')[1])) for p in coords.split()]
    predicted_bounding_boxes.append((points))

def calculate_iou(box1, box2):
    # Calculate the intersection coordinates
    x1 = max(box1[0][0], box2[0][0])
    y1 = max(box1[0][1], box2[0][1])
    x2 = min(box1[2][0], box2[2][0])
    y2 = min(box1[2][1], box2[2][1])

    # Calculate intersection area
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate areas of both boxes
    box1_area = (box1[2][0] - box1[0][0] + 1) * (box1[2][1] - box1[0][1] + 1)
    box2_area = (box2[2][0] - box2[0][0] + 1) * (box2[2][1] - box2[0][1] + 1)

    # Calculate IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou

def match_boxes(gt_boxes, pred_boxes, threshold=0.5):
    matched_pairs = []
    for gt_box in gt_boxes:
        best_iou = 0
        best_pred_box = None
        for pred_box in pred_boxes:
            iou = calculate_iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_pred_box = pred_box
        if best_iou >= threshold:
            matched_pairs.append((gt_box, best_pred_box))

    accuracy = len(matched_pairs) / len(gt_boxes)
    return accuracy, matched_pairs

# Match bounding boxes
accuracy, matched_pairs = match_boxes(gt_bounding_boxes, predicted_bounding_boxes, threshold=0.5)
print("Accuracy:", accuracy)
print("Matched pairs:", matched_pairs)

