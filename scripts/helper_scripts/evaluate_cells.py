from shapely.geometry import Polygon
import numpy as np
import os

from get_GT_cell_coords import parse_page_xml
from get_cell_coords import get_cells

def calculate_iou(polygon1, polygon2):
    """
    Calculate IoU (Intersection over Union) between two polygons.
    :param polygon1: Shapely polygon
    :param polygon2: Shapely polygon
    :return: IoU value
    """
    intersection_area = polygon1.intersection(polygon2).area
    union_area = polygon1.union(polygon2).area

    iou = intersection_area / union_area if union_area != 0 else 0.0

    return iou

def evaluate_polygons(ground_truth_polygons, predicted_polygons, threshold=0):
    """
    Evaluate predicted polygons against ground truth polygons.
    :param ground_truth_polygons: List of Shapely polygons
    :param predicted_polygons: List of Shapely polygons
    :param threshold: IoU threshold for matching polygons
    :return: Precision, Recall, IoU
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred_polygon in predicted_polygons:
        max_iou = 0
        for gt_polygon in ground_truth_polygons:
            iou = calculate_iou(pred_polygon, gt_polygon)
            if iou > max_iou:
                max_iou = iou
        if max_iou > threshold:
            true_positives += 1
        else:
            false_positives += 1

    for gt_polygon in ground_truth_polygons:
        max_iou = 0
        for pred_polygon in predicted_polygons:
            iou = calculate_iou(pred_polygon, gt_polygon)
            if iou > max_iou:
                max_iou = iou
        if max_iou <= threshold:
            false_negatives += 1

    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)
    #mean_iou = np.mean([calculate_iou(pred_polygon, gt_polygon) for pred_polygon in predicted_polygons for gt_polygon in ground_truth_polygons])
    matched_ious = [calculate_iou(pred_polygon, gt_polygon) for pred_polygon in predicted_polygons for gt_polygon in ground_truth_polygons]
    mean_iou = np.mean([iou for iou in matched_ious if iou > threshold])

    return mean_iou, precision, recall

def calculate_f1_score(precision, recall):
    """
    Calculate F1 score from precision and recall.
    :param precision: Precision value
    :param recall: Recall value
    :return: F1 score
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def find_optimal_iou_threshold(ground_truth_polygons, predicted_polygons, thresholds=np.arange(0.5, 0.6, 0.1)):
    """
    Find the IoU threshold that maximizes the F1 score.
    :param ground_truth_polygons: List of ground truth polygons
    :param predicted_polygons: List of predicted polygons
    :param thresholds: List of IoU thresholds to evaluate
    :return: Optimal IoU threshold, Optimal F1 score
    """
    best_threshold = 0.0
    best_f1_score = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_mean_iou = 0.0

    for threshold in thresholds:
        precision, recall, mean_iou = evaluate_polygons(ground_truth_polygons, predicted_polygons, threshold)
        f1_score = calculate_f1_score(precision, recall)
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_mean_iou = mean_iou

    return best_threshold, best_f1_score, best_precision, best_recall, best_mean_iou

def eval_one_image(xml_GT, xml_loghi, image_path):
    GT_cells = parse_page_xml(xml_GT)
    predicted_cells = get_cells(xml_loghi, image_path)

    return GT_cells, predicted_cells

def all_images():
    mean_iou = 0
    mean_precision = 0
    mean_recall = 0
    mean_f1 = 0
    total_images = 0

    # Define the directories
    dir1 = '/home/roderickmajoor/Desktop/Master/Thesis/GT_data'
    dir2 = '/home/roderickmajoor/Desktop/Master/Thesis/GT_data'
    dir3 = '/home/roderickmajoor/Desktop/Master/Thesis/loghi/data'

    # Get the list of subdirectories in dir1 (assuming all directories have the same subdirectories)
    subdirs = os.listdir(dir1)

    # Iterate over the subdirectories
    for subdir in subdirs:
        # Get the list of files in each subdirectory
        files1 = os.listdir(os.path.join(dir1, subdir, 'page'))
        files2 = os.listdir(os.path.join(dir2, subdir))
        files3 = os.listdir(os.path.join(dir3, subdir, 'page'))

        # Find common filenames (excluding extensions)
        filenames = set(os.path.splitext(file)[0] for file in files1) & \
                    set(os.path.splitext(file)[0] for file in files2) & \
                    set(os.path.splitext(file)[0] for file in files3)

        # Iterate over the common filenames
        mean_iou_subdir = 0
        mean_precision_subdir = 0
        mean_recall_subdir = 0
        mean_f1_subdir = 0
        total_images_subdir = 0

        print("Subdir:", subdir)
        for filename in filenames:
            # Construct the file paths
            gt_xml = os.path.join(dir1, subdir, 'page', filename + '.xml')
            jpg_file = os.path.join(dir2, subdir, filename + '.jpg')
            loghi_xml = os.path.join(dir3, subdir, 'page', filename + '.xml')

            # Process the files as needed
            ground_truth_polygons, predicted_polygons = eval_one_image(gt_xml, loghi_xml, jpg_file)

            iou_score, precision, recall = evaluate_polygons(ground_truth_polygons, predicted_polygons)
            f1 = calculate_f1_score(precision, recall)
            mean_iou += iou_score
            mean_precision += precision
            mean_recall += recall
            mean_f1 += f1
            total_images += 1

            mean_iou_subdir += iou_score
            mean_precision_subdir += precision
            mean_recall_subdir += recall
            mean_f1_subdir += f1
            total_images_subdir += 1

        mean_iou_subdir = mean_iou_subdir / total_images_subdir
        mean_precision_subdir = mean_precision_subdir / total_images_subdir
        mean_recall_subdir = mean_recall_subdir / total_images_subdir
        mean_f1_subdir = mean_f1_subdir / total_images_subdir


        print("IoU Score:", mean_iou_subdir)
        print("Precision:", mean_precision_subdir)
        print("Recall:", mean_recall_subdir)
        print("F1:", mean_f1_subdir)


    mean_iou = mean_iou / total_images
    mean_precision = mean_precision / total_images
    mean_recall = mean_recall / total_images
    mean_f1 = mean_f1 / total_images

    return mean_iou, mean_precision, mean_recall, mean_f1

mean_iou, mean_precision, mean_recall, mean_f1 = all_images()

print("Mean IoU:", mean_iou)
print("Mean Precision:", mean_precision)
print("Mean Recall:", mean_recall)
print("Mean F1 score:", mean_f1)