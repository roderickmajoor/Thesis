from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment
import numpy as np
import cv2
import os

from get_GT_cell_coords import parse_page_xml
from get_cell_coords import get_cells, find_loghi_words, highlight_words

def calculate_iou(poly1, poly2):
    # Calculate intersection over union
    return poly1.intersection(poly2).area / poly1.union(poly2).area

def eval_one_image(xml_GT, xml_loghi, image_path):
    GT_cells = parse_page_xml(xml_GT)
    predicted_cells = get_cells(xml_loghi, image_path)

    return GT_cells, predicted_cells

#xml_GT = '/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/page/WBMA00007000010.xml'
#xml_loghi = '/home/roderickmajoor/Desktop/Master/Thesis/loghi/data/55/page/WBMA00007000010.xml'
#image_path = '/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/WBMA00007000010.jpg'
def one(xml_GT, xml_loghi, image_path):

    gt_polygons, pred_polygons = eval_one_image(xml_GT, xml_loghi, image_path)

    # Assume gt_polygons and pred_polygons are your lists of ground truth and predicted polygons
    iou_matrix = [[calculate_iou(gt, pred) for pred in pred_polygons] for gt in gt_polygons]

    # Use the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)

    tp = 0
    fn = 0
    threshold = 0
    for row, col in zip(row_ind, col_ind):
        if iou_matrix[row][col] > threshold:
            tp += 1


    fn = len(gt_polygons) - tp
    fp = len(pred_polygons) - tp

    # Flatten the matrix
    matrix_array = np.array(iou_matrix)

    # Calculate mean iou, precision and recall
    iou_score = np.mean(matrix_array[matrix_array > threshold])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    #print(len(gt_polygons), len(pred_polygons))
    #print(tp, fp, fn)
    #print("Mean Matched IoU:", iou_score)
    #print("Precision:", precision)
    #print("Recall:", recall)

    return iou_score, precision, recall, f1

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
            iou_score, precision, recall, f1 = one(gt_xml, loghi_xml, jpg_file)
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

"""
image = cv2.imread(image_path)
loghi_words_dict = find_loghi_words(xml_loghi)
image = highlight_words(loghi_words_dict, image)

for value in row_ind:
    polygon = gt_polygons[value]
    # Convert Shapely polygon coordinates to NumPy array
    polygon_points = np.array(polygon.exterior.coords, dtype=np.int32)

    # Reshape the array for OpenCV
    polygon_points = polygon_points.reshape((-1, 1, 2))

    # Draw the polygon on the image
    cv2.polylines(image, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

for value in col_ind:
    polygon = pred_polygons[value]
    # Convert Shapely polygon coordinates to NumPy array
    polygon_points = np.array(polygon.exterior.coords, dtype=np.int32)

    # Reshape the array for OpenCV
    polygon_points = polygon_points.reshape((-1, 1, 2))

    # Draw the polygon on the image
    cv2.polylines(image, [polygon_points], isClosed=True, color=(255, 0, 0), thickness=2)

# Display the image
#cv2.namedWindow('Cells', cv2.WINDOW_NORMAL)
cv2.imshow("Cells", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""