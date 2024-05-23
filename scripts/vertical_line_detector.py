import cv2
import numpy as np
import math
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime

class HoughBundler:
    def __init__(self,min_distance=1000,min_angle=20):
        self.min_distance = min_distance
        self.min_angle = min_angle

    def get_orientation(self, line):
        orientation = math.atan2(abs((line[3] - line[1])), abs((line[2] - line[0])))
        return math.degrees(orientation)

    def check_is_line_different(self, line_1, groups, min_distance_to_merge, min_angle_to_merge):
        for group in groups:
            for line_2 in group:
                if self.get_distance(line_2, line_1) < min_distance_to_merge:
                    orientation_1 = self.get_orientation(line_1)
                    orientation_2 = self.get_orientation(line_2)
                    if abs(orientation_1 - orientation_2) < min_angle_to_merge:
                        group.append(line_1)
                        return False
        return True

    def distance_point_to_line(self, point, line):
        px, py = point
        x1, y1, x2, y2 = line

        def line_magnitude(x1, y1, x2, y2):
            line_magnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return line_magnitude

        lmag = line_magnitude(x1, y1, x2, y2)
        if lmag < 0.00000001:
            distance_point_to_line = 9999
            return distance_point_to_line

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (lmag * lmag)

        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = line_magnitude(px, py, x1, y1)
            iy = line_magnitude(px, py, x2, y2)
            if ix > iy:
                distance_point_to_line = iy
            else:
                distance_point_to_line = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance_point_to_line = line_magnitude(px, py, ix, iy)

        return distance_point_to_line

    def get_distance(self, a_line, b_line):
        dist1 = self.distance_point_to_line(a_line[:2], b_line)
        dist2 = self.distance_point_to_line(a_line[2:], b_line)
        dist3 = self.distance_point_to_line(b_line[:2], a_line)
        dist4 = self.distance_point_to_line(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_into_groups(self, lines):
        groups = []  # all lines groups are here
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.check_is_line_different(line_new, groups, self.min_distance, self.min_angle):
                groups.append([line_new])

        return groups

    def merge_line_segments(self, lines):
        orientation = self.get_orientation(lines[0])

        if(len(lines) == 1):
            return np.block([[lines[0][:2], lines[0][2:]]])

        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        if 45 < orientation <= 90:
            #sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            #sort by x
            points = sorted(points, key=lambda point: point[0])

        return np.block([[points[0],points[-1]]])

    def process_lines(self, lines):
        lines_horizontal  = []
        lines_vertical  = []

        for line_i in [l[0] for l in lines]:
            orientation = self.get_orientation(line_i)
            # if vertical
            if 45 < orientation <= 90:
                lines_vertical.append(line_i)
            else:
                lines_horizontal.append(line_i)

        lines_vertical  = sorted(lines_vertical , key=lambda line: line[1])
        lines_horizontal  = sorted(lines_horizontal , key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_horizontal, lines_vertical]:
            if len(i) > 0:
                groups = self.merge_lines_into_groups(i)
                merged_lines = []
                for group in groups:
                    merged_lines.append(self.merge_line_segments(group))
                merged_lines_all.extend(merged_lines)

        return np.asarray(merged_lines_all)

def vertical_line_finder(image, foreground, edges):
    left_edge, right_edge = edges
    #image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 3, 21)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    #cv2.imwrite('/home/roderickmajoor/Desktop/Master/Thesis/images/blur_image.jpg', blur)

    result = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    adaptive_threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 3)

    #cv2.namedWindow('Columns', cv2.WINDOW_NORMAL)
    #cv2.imshow('Columns', adaptive_threshold)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    kernel_dilate = np.ones((5, 1), np.uint8)
    kernel_erode = np.ones((3, 1), np.uint8)
    result = cv2.dilate(result, kernel_dilate, iterations=1)
    adaptive_threshold = cv2.erode(adaptive_threshold, kernel_erode, iterations=2)

    #cv2.imwrite('/home/roderickmajoor/Desktop/Master/Thesis/images/adaptive_threshold.jpg', adaptive_threshold)
    #cv2.imwrite('/home/roderickmajoor/Desktop/Master/Thesis/images/otsu_threshold.jpg', result)

    adaptive_threshold = cv2.subtract(adaptive_threshold, result)

    adaptive_threshold = cv2.dilate(adaptive_threshold, kernel_erode, iterations=2)
    #cv2.imwrite('/home/roderickmajoor/Desktop/Master/Thesis/images/subtract_threshold.jpg', adaptive_threshold)


    # Apply Hough line detection
    lines = cv2.HoughLinesP(adaptive_threshold, rho=1, theta=np.pi/180, threshold=100, minLineLength=500, maxLineGap=100)

    height, width = image.shape[:2]
    black_and_white_image = np.zeros((height, width), dtype=np.uint8)

    if lines is not None:
        bundler = HoughBundler(min_distance=10,min_angle=10)
        lines = bundler.process_lines(lines)

        fg_y = foreground[1]
        fg_h = foreground[3]
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate line length
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # Filter out lines with length less than or equal to 500
            if line_length > 1000 and x1 > left_edge + 10 and x1 < right_edge - 10 and x2 > left_edge + 10 and x2 < right_edge - 10:
            #    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 10)
                if x2 - x1 == 0: # Avoid division by zero
                    cv2.line(black_and_white_image, (x1, fg_y), (x1, fg_y+fg_h), (255, 255, 255), 2)
                else:
                    slope = (y2 - y1) / (x2 - x1) # Calculate slope
                    intercept = y1 - slope * x1   # Calculate intercept

                    if slope != 0:
                        # Define new y coordinates
                        y1_new = fg_y
                        y2_new = fg_y + fg_h

                        # Calculate corresponding x coordinates
                        x1_new = (y1_new - intercept) / slope
                        x2_new = (y2_new - intercept) / slope

                        if slope < -50 or slope > 50:
                        # Draw the extended line
                            cv2.line(black_and_white_image, (int(x1_new), y1_new), (int(x2_new), y2_new), (255, 255, 255), 2)
                    else: #If slope is zero
                        # Draw the extended line (vertical line)
                        pass
                        #cv2.line(black_and_white_image, (x1, fg_y), (x1, fg_y + fg_h), (255, 255, 255), 2)

    # Merge close lines and remove noise

    #cv2.namedWindow('cols', cv2.WINDOW_NORMAL)
    #cv2.imshow('cols', black_and_white_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imwrite('/home/roderickmajoor/Desktop/Master/Thesis/images/hough_lines.jpg', black_and_white_image)

    kernel = np.ones((3,3), np.uint8)
    black_and_white_image = cv2.dilate(black_and_white_image, kernel, iterations=25)
    black_and_white_image = cv2.erode(black_and_white_image, kernel, iterations=25)

    #cv2.namedWindow('cols', cv2.WINDOW_NORMAL)
    #cv2.imshow('cols', black_and_white_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    def find_middle(contour):
        # Convert contour to 2D array of points
        points = np.squeeze(contour)

        # Top left point has the smallest sum (x+y)
        top_left = points[np.argmin(np.sum(points, axis=1))]

        # Bottom right point has the largest sum (x+y)
        bottom_right = points[np.argmax(np.sum(points, axis=1))]

        # Top right point has the smallest difference (x-y)
        top_right = points[np.argmin(np.diff(points, axis=1))]

        # Bottom left point has the largest difference (x-y)
        bottom_left = points[np.argmax(np.diff(points, axis=1))]

        # Calculate the middle x-coordinate
        bottom_middle = ((bottom_left[0] + bottom_right[0]) // 2, (bottom_left[1] + bottom_right[1]) // 2)
        top_middle = ((top_left[0] + top_right[0]) // 2, (top_left[1] + top_right[1]) // 2)

        return bottom_middle, top_middle

    def draw_line_from_top_to_bottom(image, contour):
        # Get the topmost and bottommost points for the middle x-coordinate
        topmost, bottommost = find_middle(contour)

        # Draw a line from the topmost to bottommost point
        cv2.line(image, topmost, bottommost, (255, 255, 255), 2)  # Adjust thickness as needed

        return image

    # Apply houglines again to find full lines
    result = np.zeros((height, width), dtype=np.uint8)
    contours, _ = cv2.findContours(black_and_white_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        result_image = draw_line_from_top_to_bottom(result, contour)

    #lines = cv2.HoughLinesP(black_and_white_image, rho=1, theta=np.pi/180, threshold=100, minLineLength=10, maxLineGap=10)

    #if lines is not None:
    #    for line in lines:
    #        x1, y1, x2, y2 = line[0]
    #        cv2.line(result, (x1, y1), (x2, y2), (255, 255, 255), 2)

    #cv2.namedWindow('Columns', cv2.WINDOW_NORMAL)
    #cv2.imshow('Columns', result_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imwrite('/home/roderickmajoor/Desktop/Master/Thesis/images/columns_lines.jpg', result_image)


    return result_image

def group_points_by_proximity(lines, threshold):
    grouped_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            found_group = False
            for group in grouped_lines:
                group_y = group[0][0][1]  # Get the y-coordinate of the first point in the group
                if abs(y1 - group_y) <= threshold:
                    group.append(line)
                    found_group = True
                    break
            if not found_group:
                grouped_lines.append([line])
    return grouped_lines

def horizontal_line_finder(image, foreground):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    result = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    adaptive_threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12)

    kernel_dilate = np.ones((3, 3), np.uint8)
    kernel_erode = np.ones((1, 5), np.uint8)
    result = cv2.dilate(result, kernel_dilate, iterations=1)
    adaptive_threshold = cv2.erode(adaptive_threshold, kernel_erode, iterations=2)
    adaptive_threshold = cv2.subtract(adaptive_threshold, result)
    adaptive_threshold = cv2.dilate(adaptive_threshold, kernel_erode, iterations=2)

    height, width = image.shape[:2]
    black_and_white_image = np.zeros((height, width), dtype=np.uint8)

    # Apply Hough line detection
    lines = cv2.HoughLinesP(adaptive_threshold, rho=1, theta=np.pi/180, threshold=100, minLineLength=0.25*width, maxLineGap=300)

    fg_x = foreground[0]
    fg_w = foreground[2]

    #left_most_x = width
    #right_most_x = 0

    threshold = 100  # Adjust this threshold as needed
    grouped_lines = group_points_by_proximity(lines, threshold)

    if grouped_lines is not None:
        for group in grouped_lines:
            left_most_x = width
            right_most_x = 0
            for line in group:
                x1, y1, x2, y2 = line[0]
                if x1 < left_most_x:
                    left_most_x = x1
                    left_most_y = y1
                if x2 > right_most_x:
                    right_most_x = x2
                    right_most_y = y2

            cv2.line(black_and_white_image, (fg_x, left_most_y), (fg_x + fg_w, right_most_y), (255, 255, 255), 2)  # Draw green lines over detected table columns


    #if lines is not None:
    #    for line in lines:
    #        x1, y1, x2, y2 = line[0]
    #        if x1 < left_most_x:
    #            left_most_x = x1
    #            left_most_y = y1
    #        if x2 > right_most_x:
    #            right_most_x = x2
    #            right_most_y = y2

    #cv2.line(black_and_white_image, (fg_x, left_most_y), (fg_x + fg_w, right_most_y), (255, 255, 255), 2)  # Draw green lines over detected table columns

    return black_and_white_image

def all_lines_finder(image, original_image):
    lines = cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=100, minLineLength=10, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green lines over detected table columns

    return original_image, lines

def foreground_extractor(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3,3), 0)

    # Apply thresholding to create a binary image
    binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Invert the binary image (optional, depends on your image)
    binary = cv2.bitwise_not(binary)

    kernel = np.ones((10,10), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=2)
    binary = cv2.dilate(binary, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate the corner points
    top_left = (x, y)
    top_right = (x + w, y)
    bottom_left = (x, y + h)
    bottom_right = (x + w, y + h)

    height, width = image.shape[:2]
    black_and_white_image = np.zeros((height, width), dtype=np.uint8)

    # Draw the rectangle
    cv2.rectangle(black_and_white_image, top_left, bottom_right, (255, 255, 255), cv2.FILLED)

    return black_and_white_image, (x,y,w,h)

def process_images_in_folder(folder_path):

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            fg_image, foreground = foreground_extractor(image.copy())
            vertical_lines = vertical_line_finder(image.copy(), foreground)
            horizontal_lines = horizontal_line_finder(image.copy(), foreground)
            all_lines, lines = all_lines_finder(cv2.add(vertical_lines, horizontal_lines), image.copy())
            cells = cv2.subtract(fg_image, cv2.add(vertical_lines, horizontal_lines))

            #cv2.namedWindow(image_path, cv2.WINDOW_NORMAL)
            #cv2.imshow(image_path, cells)
            #cv2.waitKey(0)

def annotaions():
    # Example usage
    folder_path = "/media/roderickmajoor/TREKSTOR/Train/images/"

    # Function to convert contour to COCO segmentation format
    def contour_to_segmentation(contour):
        segmentation = []
        for point in contour:
            if isinstance(point, tuple):
                segmentation.append(float(point[0]))
                segmentation.append(float(point[1]))
        return [segmentation]

    # Initialize COCO annotation structure
    coco_annotation = {
        "info": {
            "year": 2024,
            "version": "1.0",
            "description": "VIA project exported to COCO format using VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via/)",
            "contributor": "",
            "url": "http://www.robots.ox.ac.uk/~vgg/software/via/",
            "date_created": str(datetime.now())
        },
        "images": [],
        "annotations": [],
        "licenses": [{"id": 0, "name": "Unknown License", "url": ""}],
        "categories": [{"supercategory": "type", "id": 1, "name": "Table area"}]
    }

    # Counter for image IDs
    image_id_counter = 1

    # Loop over images in folder
    for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)

                fg_image, foreground = foreground_extractor(image.copy())
                vertical_lines = vertical_line_finder(image.copy(), foreground)
                horizontal_lines = horizontal_line_finder(image.copy(), foreground)
                all_lines, lines = all_lines_finder(cv2.add(vertical_lines, horizontal_lines), image.copy())
                cells = cv2.subtract(fg_image, cv2.add(vertical_lines, horizontal_lines))

            # Find contours in the binary image
            contours, _ = cv2.findContours(cells, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Loop over contours
            for contour in contours:
                # Convert contour to COCO segmentation format
                segmentation = contour_to_segmentation(contour.squeeze())

                # Calculate bounding box
                x, y, w, h = cv2.boundingRect(contour)
                bbox = [x, y, w, h]

                # Add annotation to COCO structure
                annotation = {
                    "segmentation": segmentation,
                    "area": cv2.contourArea(contour),
                    "bbox": bbox,
                    "iscrowd": 0,
                    "id": len(coco_annotation["annotations"]) + 1,
                    "image_id": image_id_counter,
                    "category_id": 1  # Table area category ID
                }
                coco_annotation["annotations"].append(annotation)

            # Add image information to COCO structure
            coco_annotation["images"].append({
                "id": image_id_counter,
                "width": image.shape[1],
                "height": image.shape[0],
                "file_name": filename,
                "license": 0,
                "flickr_url": filename,
                "coco_url": filename,
                "date_captured": ""
            })

            # Increment image ID counter
            image_id_counter += 1

    # Save COCO annotation to JSON file
    output_path = "/home/roderickmajoor/Desktop/output.json"
    with open(output_path, "w") as f:
        json.dump(coco_annotation, f)

    print("Annotation saved to", output_path)

#process_images_in_folder("/home/roderickmajoor/Desktop/Master/Thesis/Train/")
#cv2.destroyAllWindows()
#annotaions()

def process_image_xml(image_path):
    # Read image
    image = cv2.imread(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Apply image processing steps
    fg_image, foreground = foreground_extractor(image.copy())
    vertical_lines = vertical_line_finder(image.copy(), foreground)
    horizontal_lines = horizontal_line_finder(image.copy(), foreground)
    #cells = cv2.subtract(fg_image, cv2.add(vertical_lines, horizontal_lines)) # DIT BEPALEN!
    cells = cv2.subtract(fg_image, vertical_lines)

    # Find contours in the binary image
    contours, _ = cv2.findContours(cells, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create XML tree
    root = ET.Element("PcGts")
    metadata = ET.SubElement(root, "Metadata")
    page = ET.SubElement(root, "Page")
    page.set("imageFilename", image_name + ".jpg")

    # Loop over contours and save bounding boxes
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        region = ET.SubElement(page, "TextRegion")
        region.set("id", f"r_{i}")
        coords = ET.SubElement(region, "Coords")
        coords.set("points", f"{x},{y} {x+w},{y} {x+w},{y+h} {x},{y+h}")

    # Write PageXML file
    page_xml_path = os.path.join(os.path.dirname(image_path), "page", f"{image_name}_columns_found.xml")
    tree = ET.ElementTree(root)
    tree.write(page_xml_path)

def process_images_in_directory(directory):
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(subdir, file)
                process_image_xml(image_path)

# Directory containing the images
#images_directory = "/home/roderickmajoor/Desktop/Master/Thesis/GT_data/"

# Process images in the directory
#process_images_in_directory(images_directory)