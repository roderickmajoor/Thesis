# This script is used to extract the column lines for the rule-based method.

import cv2
import numpy as np
import math
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime

class HoughBundler:
    def __init__(self, min_distance=1000, min_angle=20):
        """
        Initialize the HoughBundler with minimum distance and angle for merging lines.
        """
        self.min_distance = min_distance
        self.min_angle = min_angle

    def get_orientation(self, line):
        """
        Calculate the orientation of a line.

        Parameters:
        - line: A line represented by its endpoints [x1, y1, x2, y2].

        Returns:
        - Orientation angle in degrees.
        """
        orientation = math.atan2(abs((line[3] - line[1])), abs((line[2] - line[0])))
        return math.degrees(orientation)

    def check_is_line_different(self, line_1, groups, min_distance_to_merge, min_angle_to_merge):
        """
        Check if a line is different from lines in existing groups.

        Parameters:
        - line_1: The line to check.
        - groups: Existing groups of lines.
        - min_distance_to_merge: Minimum distance to consider merging.
        - min_angle_to_merge: Minimum angle difference to consider merging.

        Returns:
        - True if the line is different, False otherwise.
        """
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
        """
        Calculate the distance from a point to a line.

        Parameters:
        - point: The point (x, y).
        - line: The line represented by its endpoints [x1, y1, x2, y2].

        Returns:
        - Distance from the point to the line.
        """
        px, py = point
        x1, y1, x2, y2 = line

        def line_magnitude(x1, y1, x2, y2):
            """
            Calculate the magnitude (length) of a line.
            """
            line_magnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return line_magnitude

        lmag = line_magnitude(x1, y1, x2, y2)
        if lmag < 0.00000001:
            distance_point_to_line = 9999
            return distance_point_to_line

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (lmag * lmag)

        if (u < 0.00001) or (u > 1):
            # Closest point does not fall within the line segment, take the shorter distance to an endpoint
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
        """
        Calculate the minimum distance between two lines.

        Parameters:
        - a_line: The first line [x1, y1, x2, y2].
        - b_line: The second line [x1, y1, x2, y2].

        Returns:
        - The minimum distance between the two lines.
        """
        dist1 = self.distance_point_to_line(a_line[:2], b_line)
        dist2 = self.distance_point_to_line(a_line[2:], b_line)
        dist3 = self.distance_point_to_line(b_line[:2], a_line)
        dist4 = self.distance_point_to_line(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_into_groups(self, lines):
        """
        Merge lines into groups based on distance and orientation.

        Parameters:
        - lines: List of lines to merge.

        Returns:
        - Groups of merged lines.
        """
        groups = []  # all lines groups are here
        groups.append([lines[0]])  # first line will create new group every time
        for line_new in lines[1:]:
            if self.check_is_line_different(line_new, groups, self.min_distance, self.min_angle):
                groups.append([line_new])

        return groups

    def merge_line_segments(self, lines):
        """
        Merge line segments in a group into a single line segment.

        Parameters:
        - lines: Group of lines to merge.

        Returns:
        - Merged line segment.
        """
        orientation = self.get_orientation(lines[0])

        if len(lines) == 1:
            return np.block([[lines[0][:2], lines[0][2:]]])

        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        if 45 < orientation <= 90:
            # Sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            # Sort by x
            points = sorted(points, key=lambda point: point[0])

        return np.block([[points[0], points[-1]]])

    def process_lines(self, lines):
        """
        Process lines to separate into horizontal and vertical lines and merge them.

        Parameters:
        - lines: List of lines to process.

        Returns:
        - Array of merged lines.
        """
        lines_horizontal = []
        lines_vertical = []

        for line_i in [l[0] for l in lines]:
            orientation = self.get_orientation(line_i)
            if 45 < orientation <= 90:
                lines_vertical.append(line_i)
            else:
                lines_horizontal.append(line_i)

        lines_vertical = sorted(lines_vertical, key=lambda line: line[1])
        lines_horizontal = sorted(lines_horizontal, key=lambda line: line[0])
        merged_lines_all = []

        for i in [lines_horizontal, lines_vertical]:
            if len(i) > 0:
                groups = self.merge_lines_into_groups(i)
                merged_lines = []
                for group in groups:
                    merged_lines.append(self.merge_line_segments(group))
                merged_lines_all.extend(merged_lines)

        return np.asarray(merged_lines_all)

def vertical_line_finder(image, foreground, edges):
    """
    Find vertical lines in an image using Hough Transform.

    Parameters:
    - image: The input image.
    - foreground: Tuple containing the coordinates and size of the foreground region.
    - edges: Tuple containing the left and right edges of the area of interest.

    Returns:
    - Image with detected vertical lines.
    """
    left_edge, right_edge = edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    result = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    adaptive_threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 3)

    kernel_dilate = np.ones((5, 1), np.uint8)
    kernel_erode = np.ones((3, 1), np.uint8)
    result = cv2.dilate(result, kernel_dilate, iterations=1)
    adaptive_threshold = cv2.erode(adaptive_threshold, kernel_erode, iterations=2)

    adaptive_threshold = cv2.subtract(adaptive_threshold, result)
    adaptive_threshold = cv2.dilate(adaptive_threshold, kernel_erode, iterations=2)

    # Apply Hough line detection
    lines = cv2.HoughLinesP(adaptive_threshold, rho=1, theta=np.pi/180, threshold=100, minLineLength=500, maxLineGap=100)

    height, width = image.shape[:2]
    black_and_white_image = np.zeros((height, width), dtype=np.uint8)

    if lines is not None:
        bundler = HoughBundler(min_distance=10, min_angle=10)
        lines = bundler.process_lines(lines)

        fg_y = foreground[1]
        fg_h = foreground[3]
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if line_length > 1000 and x1 > left_edge + 10 and x1 < right_edge - 10 and x2 > left_edge + 10 and x2 < right_edge - 10:
                if x2 - x1 == 0:
                    # If the line is vertical
                    cv2.line(black_and_white_image, (x1, fg_y), (x1, fg_y + fg_h), (255, 255, 255), 2)
                else:
                    slope = (y2 - y1) / (x2 - x1)  # Calculate slope
                    intercept = y1 - slope * x1  # Calculate intercept

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
                    else:
                        # If the slope is zero, do nothing
                        pass

    # Merge close lines and remove noise
    kernel = np.ones((3, 3), np.uint8)
    black_and_white_image = cv2.dilate(black_and_white_image, kernel, iterations=25)
    black_and_white_image = cv2.erode(black_and_white_image, kernel, iterations=25)

    def find_middle(contour):
        """
        Find the middle x-coordinates of the top and bottom of a contour.

        Parameters:
        - contour: The contour to process.

        Returns:
        - Tuple of bottom middle and top middle points.
        """
        points = np.squeeze(contour)

        top_left = points[np.argmin(np.sum(points, axis=1))]
        bottom_right = points[np.argmax(np.sum(points, axis=1))]
        top_right = points[np.argmin(np.diff(points, axis=1))]
        bottom_left = points[np.argmax(np.diff(points, axis=1))]

        bottom_middle = ((bottom_left[0] + bottom_right[0]) // 2, (bottom_left[1] + bottom_right[1]) // 2)
        top_middle = ((top_left[0] + top_right[0]) // 2, (top_left[1] + top_right[1]) // 2)

        return bottom_middle, top_middle

    def draw_line_from_top_to_bottom(image, contour):
        """
        Draw a line from the top to the bottom middle of a contour.

        Parameters:
        - image: The image to draw on.
        - contour: The contour to process.

        Returns:
        - Image with the drawn line.
        """
        topmost, bottommost = find_middle(contour)
        cv2.line(image, topmost, bottommost, (255, 255, 255), 2)
        return image

    # Apply HoughLines again to find full lines
    result = np.zeros((height, width), dtype=np.uint8)
    contours, _ = cv2.findContours(black_and_white_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        for contour in contours:
            result_image = draw_line_from_top_to_bottom(result, contour)
    else:
        result_image = result

    return result_image

def foreground_extractor(image):
    """
    Extract the foreground of an image by finding the largest contour.

    Parameters:
    - image: The input image.

    Returns:
    - Black and white image with the foreground.
    - Tuple containing the bounding box of the largest contour.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    binary = cv2.bitwise_not(binary)

    kernel = np.ones((10, 10), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=2)
    binary = cv2.dilate(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    top_left = (x, y)
    top_right = (x + w, y)
    bottom_left = (x, y + h)
    bottom_right = (x + w, y + h)

    height, width = image.shape[:2]
    black_and_white_image = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(black_and_white_image, top_left, bottom_right, (255, 255, 255), cv2.FILLED)

    return black_and_white_image, (x, y, w, h)
