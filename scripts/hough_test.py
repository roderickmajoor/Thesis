import math
import numpy as np
import cv2

class HoughBundler:
    def __init__(self,min_distance=5,min_angle=2):
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

def preprocess_image(image):
    #image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 3, 21)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    result = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    test = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)[1]
    # Create a mask where white regions of the thresholded image are non-zero
    #mask = cv2.threshold(result, 200, 255, cv2.THRESH_BINARY)[1]

    #Apply the mask to the original image
    #result = cv2.bitwise_or(image, cv2.merge([result, result, result]))
    #result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    #result = cv2.GaussianBlur(result, (3,3), 0)

    kernel_dilate = np.ones((3, 3), np.uint8)
    result = cv2.dilate(result, kernel_dilate, iterations=1)

    #result = cv2.threshold(result, 190, 255, cv2.THRESH_BINARY)[1]
    result = cv2.subtract(test, result)

    # Define the kernel for erosion and dilation
    kernel_erode = np.ones((100, 1), np.uint8)
    kernel_dilate = np.ones((100, 1), np.uint8)

    # Erode the black regions
    #result = cv2.dilate(result, kernel_dilate, iterations=1)

    # Dilate the eroded image
    #result = cv2.dilate(result, kernel_dilate, iterations=1)
    #result = cv2.erode(result, kernel_erode, iterations=3)
    #result = cv2.dilate(result, kernel_dilate, iterations=2)


    # Display the result in a resizable window
    #cv2.namedWindow('Vertical Lines Detected', cv2.WINDOW_NORMAL)
    #cv2.imshow('Vertical Lines Detected', result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return result

# Usage:
image = cv2.imread("/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/WBMA00007000010.jpg")
preprocessed_image = preprocess_image(image.copy())

lines = cv2.HoughLinesP(preprocessed_image, 1, np.pi / 180, 50, None, 50, 10)
bundler = HoughBundler(min_distance=10,min_angle=5)
lines = bundler.process_lines(lines)

if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate the length of the line
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Calculate the orientation of the line
            orientation = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi

            # Filter lines based on length and orientation
            if length > 500 and abs(orientation) > 80:
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.namedWindow('Lines Image', cv2.WINDOW_NORMAL)
cv2.imshow('Lines Image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()