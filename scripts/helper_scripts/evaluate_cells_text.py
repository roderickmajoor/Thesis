import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import Levenshtein
from collections import Counter
from shapely.geometry import Polygon
from shapely.validation import make_valid
from scipy.optimize import linear_sum_assignment


from get_cell_coords import get_cells

ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

def calculate_iou(poly1, poly2):
    # Calculate intersection over union
    return poly1.intersection(poly2).area / poly1.union(poly2).area

def get_leftmost_x(polygon):
    """Returns the x-coordinate of the left-most point of the polygon."""
    return min(point[0] for point in polygon.exterior.coords)

def text_match(text1, text2, threshold):
    """Determine if two texts match based on Levenshtein distance threshold."""
    lev_distance = Levenshtein.distance(text1, text2)
    max_len = max(len(text1), len(text2))
    return lev_distance / max_len <= threshold, lev_distance

def one_image_data(root_gt, root_loghi, predicted_cells):

    table_cells_gt = root_gt.findall('.//page:TableCell', ns)
    text_regions_gt = root_gt.findall('.//page:TextRegion', ns)
    all_regions_gt = text_regions_gt + table_cells_gt

    words_loghi = root_loghi.findall('.//page:Word', ns)

    # Frequently occuring substituions made by htr system where number -> char
    replacements = {
        'a': '1', 'n': '7', 's': '8', 'r': '2', 'i': '1', 't': '1', 'o': '0', 'g': '9',
        'e': '1', 'd': '1', 'ƒ': '1', 'p': '7', 'S': '8', 'k': '3', 'R': '7', 'u': '1',
        'v': '1', 'l': '1', 'h': '2', 'f': '1', 'B': '3', 'b': '6', 'm': '9', 'w': '1',
        'y': '9', 'c': '1', 'C': '1', 'I': '1', 'N': '1', 'E': '6', 'z': '3', 'q': '9'
    }

    # Initialize dictionaries to store region information
    gt_regions_dict = {}
    loghi_words_dict = {}

    # Compare words in original data with table cells in ground truth
    for i, region_gt in enumerate(table_cells_gt):
        coords_gt = region_gt.find('page:Coords', ns).attrib['points']
        text_gt = region_gt.find('.//page:TextEquiv/page:Unicode', ns).text if region_gt.find('.//page:TextEquiv/page:Unicode', ns) is not None else ""

        if text_gt is None or text_gt == '':
            continue

        points = [tuple(map(int, point.split(','))) for point in coords_gt.split()]
        polygon = Polygon(points)

        gt_regions_dict[i] = {'coords': polygon, 'text': text_gt}


    for j, word_loghi in enumerate(words_loghi):
        coords_loghi = word_loghi.find('page:Coords', ns).attrib['points']
        text_loghi = word_loghi.find('page:TextEquiv/page:Unicode', ns).text
        #loghi_words_dict[j] = {'coords': coords_loghi, 'text': text_loghi}
        points = [tuple(map(int, point.split(','))) for point in coords_loghi.split()]
        polygon = Polygon(points)

        # Check if text contains letters or numbers
        if any(c.isalnum() for c in text_loghi):
            # Remove characters that are not letters or numbers
            cleaned_text = ''.join(c for c in text_loghi if c.isalnum())

            # Replace specific characters if the text contains both letters and numbers
            if any(c.isdigit() for c in cleaned_text) and any(c.isalpha() for c in cleaned_text) or len(cleaned_text) == 1:
                for key, value in replacements.items():
                    cleaned_text = cleaned_text.replace(key, value)

            # Check if cleaned text is a single character and not a number
            if len(cleaned_text) > 1 or cleaned_text.isdigit():
                loghi_words_dict[j] = {'coords': polygon, 'text': cleaned_text}

    # Assume gt_polygons and pred_polygons are your lists of ground truth and predicted polygons
    iou_matrix = [[calculate_iou(gt['coords'], pred) for pred in predicted_cells] for gt in gt_regions_dict.values()]

    # Use the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)


    predicted_cell_dict = {}
    for row, col in zip(row_ind, col_ind):
        if iou_matrix[row][col] >= 0.25:
            predicted_cell_dict[col] = {'coords': predicted_cells[col], 'text': ''}

    """
    # Sorting the dictionary
    sorted_loghi_words_list = sorted(loghi_words_dict.items(), key=lambda item: get_leftmost_x(item[1]['coords']))

    # Converting back to a dictionary
    sorted_loghi_words_dict = {k: v for k, v in sorted_loghi_words_list}

    for polygon_key, polygon_data in sorted_loghi_words_dict.items():
        # Initialize variables to keep track of the highest IoU and corresponding polygon
        highest_iou = 0
        matching_polygon = None

        # Iterate over polygons in predicted_cells
        for polygon_key1, polygon_data1 in predicted_cell_dict.items():
            iou = calculate_iou(make_valid(polygon_data['coords']), make_valid(polygon_data1['coords']))

            # Update highest IoU and corresponding polygon if needed
            if iou > highest_iou:
                highest_iou = iou
                matching_polygon = polygon_key1
                #matching_polygon = polygon_data['text']

                #predicted_cell_dict[polygon_key1]['text'] += matching_polygon if predicted_cell_dict[polygon_key1]['text'] == '' else ' ' + matching_polygon

        if highest_iou > 0:
            predicted_cell_dict[matching_polygon]['text'] += polygon_data['text'] if predicted_cell_dict[matching_polygon]['text'] == '' else ' ' + polygon_data['text']
    """

    # Iterate over polygons in predicted_cells
    for polygon_key1, polygon_data1 in predicted_cell_dict.items():
        highest_iou = 0
        matching_polygon = None

        for polygon_key, polygon_data in loghi_words_dict.items():
            iou = calculate_iou(make_valid(polygon_data['coords']), make_valid(polygon_data1['coords']))

            # Update highest IoU and corresponding polygon if needed
            if iou > highest_iou:
                highest_iou = iou
                matching_polygon = polygon_data['text']

        if highest_iou > 0:
            predicted_cell_dict[polygon_key1]['text'] += matching_polygon


    text_matches_0 = 0
    text_matches_25 = 0
    text_matches_50 = 0
    text_matches_75 = 0
    total_words = 0
    total_edit_distance = 0
    len_gt_text = 0

    for row, col in zip(row_ind, col_ind):
        if iou_matrix[row][col] >= 0.25:
            #print(list(gt_regions_dict.values())[row]['text']) #Matched GT words
            #print(predicted_cell_dict[col]['text']) #Matched Predicted words
            total_words += 1
            total_edit_distance += text_match(list(gt_regions_dict.values())[row]['text'], predicted_cell_dict[col]['text'], 0)[1]
            len_gt_text += len(list(gt_regions_dict.values())[row]['text'])
            if text_match(list(gt_regions_dict.values())[row]['text'], predicted_cell_dict[col]['text'], 0.25)[0]:
                text_matches_25 += 1
            if text_match(list(gt_regions_dict.values())[row]['text'], predicted_cell_dict[col]['text'], 0.5)[0]:
                text_matches_50 += 1
            if text_match(list(gt_regions_dict.values())[row]['text'], predicted_cell_dict[col]['text'], 0.75)[0]:
                text_matches_75 += 1
            if text_match(list(gt_regions_dict.values())[row]['text'], predicted_cell_dict[col]['text'], 0)[0]:
                text_matches_0 += 1

    acc_0 = text_matches_0 / total_words
    acc_25 = text_matches_25 / total_words
    acc_50 = text_matches_50 / total_words
    acc_75 = text_matches_75 / total_words
    edit_distance = total_edit_distance / len_gt_text

    return acc_0, acc_25, acc_50, acc_75, edit_distance

def all_image_data():
    # Define the directories
    dir1 = '/home/roderickmajoor/Desktop/Master/Thesis/GT_data'
    dir2 = '/home/roderickmajoor/Desktop/Master/Thesis/GT_data'
    dir3 = '/home/roderickmajoor/Desktop/Master/Thesis/loghi/data'

    total_acc_0 = 0
    total_acc_25 = 0
    total_acc_50 = 0
    total_acc_75 = 0
    total_images = 0
    total_edit_distance = 0

    # Get the list of subdirectories in dir1 (assuming all directories have the same subdirectories)
    subdirs = os.listdir(dir1)

    # Iterate over the subdirectories
    for subdir in subdirs:
        print("subdir:", subdir)
        # Get the list of files in each subdirectory
        files1 = os.listdir(os.path.join(dir1, subdir, 'page'))
        files2 = os.listdir(os.path.join(dir2, subdir))
        files3 = os.listdir(os.path.join(dir3, subdir, 'page'))

        # Find common filenames (excluding extensions)
        filenames = set(os.path.splitext(file)[0] for file in files1) & \
                    set(os.path.splitext(file)[0] for file in files2) & \
                    set(os.path.splitext(file)[0] for file in files3)

        for filename in filenames:
                # Construct the file paths
                gt_xml = os.path.join(dir1, subdir, 'page', filename + '.xml')
                jpg_file = os.path.join(dir2, subdir, filename + '.jpg')
                loghi_xml = os.path.join(dir3, subdir, 'page', filename + '.xml')

                # Parse loghi file
                tree_loghi = ET.parse(loghi_xml)
                root_loghi = tree_loghi.getroot()

                # Get predicted cells
                predicted_cells = get_cells(loghi_xml, jpg_file)

                # Parse GT File
                tree_gt = ET.parse(gt_xml)
                root_gt = tree_gt.getroot()

                acc_0, acc_25, acc_50, acc_75, edit_distance = one_image_data(root_gt, root_loghi, predicted_cells)

                total_images += 1
                total_acc_0 += acc_0
                total_acc_25 += acc_25
                total_acc_50 += acc_50
                total_acc_75 += acc_75
                total_edit_distance += edit_distance

    mean_acc_0 = total_acc_0 / total_images
    mean_acc_25 = total_acc_25 / total_images
    mean_acc_50 = total_acc_50 / total_images
    mean_acc_75 = total_acc_75 / total_images
    cer = total_edit_distance / total_images

    return mean_acc_0, mean_acc_25, mean_acc_50, mean_acc_75, cer

mean_acc_0, mean_acc_25, mean_acc_50, mean_acc_75, cer = all_image_data()

print("Mean Acc Levensthein threshold 0: ", mean_acc_0)
print("Mean Acc Levensthein threshold 0.25: ", mean_acc_25)
print("Mean Acc Levensthein threshold 0.50: ", mean_acc_50)
print("Mean Acc Levensthein threshold 0.75: ", mean_acc_75)
print("CER: ", cer)