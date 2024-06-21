# The functions in this script are used in the other files to make the excel table and create annotations.
# It is used to extract the cell coords from the rule-based method.
# It also contains some functions that can help in visualizing what is happening, such as drawing the cells on the page.

import cv2
import numpy as np
import pandas as pd

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from postprocess import find_loghi_words, find_loghi_textlines
from vertical_line_detector import foreground_extractor, vertical_line_finder

def parse_coords(coords):
    """
    Parse the coordinates string into a list of points.

    Parameters:
    - coords: String containing coordinates in the format 'x1,y1 x2,y2 ...'.

    Returns:
    - List of tuples representing the points.
    """
    points = []
    for point_str in coords.split():
        x, y = map(int, point_str.split(','))
        points.append((x, y))
    return points

def highlight_words(loghi_words_dict, image):
    """
    Highlight words by drawing their bounding boxes on the image.

    Parameters:
    - loghi_words_dict: Dictionary containing word data.
    - image: Image on which to draw the bounding boxes.

    Returns:
    - Image with highlighted words.
    """
    for word_data in loghi_words_dict.values():
        coords = word_data['coords']
        points = parse_coords(coords)
        # Convert coordinates to numpy array and reshape for drawing
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        # Draw filled polygon on the mask
        cv2.fillPoly(image, [pts], (255, 255, 255))

    return image

def get_row_boxes(loghi_words_dict, middle):
    """
    Group words into rows based on their coordinates and the middle line.

    Parameters:
    - loghi_words_dict: Dictionary containing word data.
    - middle: Tuple containing x-coordinate and width of the middle line.

    Returns:
    - Two lists of polygons representing left and right row boxes.
    """
    boxes = []
    x_middle, w_middle = middle

    for word_data in loghi_words_dict.values():
        coords = word_data['coords']
        points = parse_coords(coords)
        # Convert coordinates to numpy array
        pts = np.array(points, np.int32)
        polygon = Polygon(pts)
        boxes.append(polygon)

    # Separate boxes into two groups: left and right
    left_boxes = [box for box in boxes if max(box.exterior.coords, key=lambda x: x[0])[0] < x_middle]
    right_boxes = [box for box in boxes if min(box.exterior.coords, key=lambda x: x[0])[0] >= x_middle + w_middle]

    def get_rows(boxes):
        """
        Group boxes with similar y-coordinates into rows.

        Parameters:
        - boxes: List of polygons.

        Returns:
        - List of combined row boxes as polygons.
        """
        # Sort boxes by x-coordinate
        boxes.sort(key=lambda box: box.centroid.x)

        rows = []  # A list containing lists of bounding boxes of words
        row_y_values = []  # A list containing tuples (y_min, y_max) for each row
        for box in boxes:
            min_y_box = min(box.exterior.coords, key=lambda x: x[1])[1]
            max_y_box = max(box.exterior.coords, key=lambda x: x[1])[1]

            if row_y_values:
                curr_best = np.inf
                best_row = len(row_y_values)
                for i, y_values in enumerate(row_y_values):
                    if (max_y_box > y_values[0] and max_y_box < y_values[1]) or (y_values[1] > min_y_box and y_values[0] < max_y_box):
                        if abs(max_y_box - y_values[1]) + abs(min_y_box - y_values[0]) < curr_best:
                            best_row = i
                            curr_best = abs(max_y_box - y_values[1]) + abs(min_y_box - y_values[0])
                if best_row == len(row_y_values):
                    rows.append([box.buffer(0)])
                    row_y_values.append((min_y_box, max_y_box))
                else:
                    rows[best_row].append(box.buffer(0))
                    row_y_values[best_row] = (min_y_box, max_y_box)
            else:
                rows.append([box.buffer(0)])
                row_y_values.append((min_y_box, max_y_box))

        # Create row bounding boxes
        row_boxes = []
        for row in rows:
            if row:
                # Use unary_union to combine the polygons
                combined_polygon = unary_union(row)
                row_boxes.append(combined_polygon)

        return row_boxes

    # Get row boxes for left and right groups
    left_row_boxes = get_rows(left_boxes)
    right_row_boxes = get_rows(right_boxes)

    return left_row_boxes, right_row_boxes

def multipolygon_to_contours(multipolygon):
    """
    Convert a Shapely MultiPolygon to OpenCV contours.

    Parameters:
    - multipolygon: A Shapely MultiPolygon.

    Returns:
    - List of contours as numpy arrays.
    """
    contours = []
    for polygon in multipolygon:
        contour = np.array(polygon.exterior.coords, dtype=np.int32)
        contours.append(contour)
    return contours

def extend_polygon_to_edges(polygon, left_edge, right_edge):
    """
    Extend the vertices of a polygon to specified left and right edges.

    Parameters:
    - polygon: Polygon vertices as a numpy array.
    - left_edge: The left edge to extend to.
    - right_edge: The right edge to extend to.

    Returns:
    - Extended polygon vertices as a numpy array.
    """
    extended_polygon = []
    for vertex in polygon:
        x_coord = vertex[0]
        # Extend to the left edge
        left_extended_vertex = [min(x_coord, left_edge), vertex[1]]
        # Extend to the right edge
        right_extended_vertex = [max(x_coord, right_edge), vertex[1]]
        extended_polygon.append(left_extended_vertex)
        extended_polygon.append(right_extended_vertex)

    return np.array(extended_polygon)

def get_row_coords(image, left_boxes, right_boxes, middle, edges):
    """
    Get coordinates of rows by extending row bounding boxes to the edges.

    Parameters:
    - image: Image used for dimensions.
    - left_boxes: List of left row bounding boxes.
    - right_boxes: List of right row bounding boxes.
    - middle: Tuple containing x-coordinate and width of the middle line.
    - edges: Tuple containing leftmost and rightmost edges.

    Returns:
    - List of row coordinates as numpy arrays.
    """
    x_middle, w_middle = middle
    leftmost, rightmost = edges
    row_coords = []
    all_boxes = left_boxes + right_boxes

    for box in all_boxes:
        if isinstance(box, MultiPolygon):
            contours = multipolygon_to_contours(box.geoms)
            combined_contour = np.concatenate(contours)

            if box in left_boxes:
                extended_contour = extend_polygon_to_edges(combined_contour, left_edge=leftmost, right_edge=x_middle)
            else:
                extended_contour = extend_polygon_to_edges(combined_contour, left_edge=x_middle + w_middle, right_edge=rightmost)

            # Approximate contours to convex hulls
            extended_contour = cv2.convexHull(extended_contour)
        else:
            points = np.array(box.exterior.coords, dtype=np.int32)

            if box in left_boxes:
                extended_contour = extend_polygon_to_edges(points, left_edge=leftmost, right_edge=x_middle)
            else:
                extended_contour = extend_polygon_to_edges(points, left_edge=x_middle + w_middle, right_edge=rightmost)

            extended_contour = cv2.convexHull(extended_contour)

        row_coords.append(extended_contour)

    return row_coords

def draw_rows(image, left_boxes, right_boxes, middle):
    """
    Draw row bounding boxes on the image.

    Parameters:
    - image: Image on which to draw the bounding boxes.
    - left_boxes: List of left row bounding boxes.
    - right_boxes: List of right row bounding boxes.
    - middle: Tuple containing x-coordinate and width of the middle line.

    Returns:
    - Image with drawn row bounding boxes.
    """
    x_middle, w_middle = middle
    # Create a black mask of the same size as the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(image)

    all_boxes = left_boxes + right_boxes

    for box in all_boxes:
        if isinstance(box, MultiPolygon):
            contours = multipolygon_to_contours(box.geoms)
            combined_contour = np.concatenate(contours)

            if box in left_boxes:
                extended_contour = extend_polygon_to_edges(combined_contour, left_edge=0, right_edge=x_middle)
            else:
                extended_contour = extend_polygon_to_edges(combined_contour, left_edge=x_middle + w_middle, right_edge=image.shape[1])

            # Approximate contours to convex hulls
            extended_contour = cv2.convexHull(extended_contour)
        else:
            points = np.array(box.exterior.coords, dtype=np.int32)

            if box in left_boxes:
                extended_contour = extend_polygon_to_edges(points, left_edge=0, right_edge=x_middle)
            else:
                extended_contour = extend_polygon_to_edges(points, left_edge=x_middle + w_middle, right_edge=image.shape[1])

            extended_contour = cv2.convexHull(extended_contour)

        cv2.polylines(mask, [extended_contour], isClosed=True, color=(255, 255, 255), thickness=2)

    return mask

def find_middle(image):
    """
    Find the middle line of the image based on contours.

    Parameters:
    - image: Binary image to find contours in.

    Returns:
    - Tuple containing x-coordinate and width of the middle line.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on x-coordinate
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # Get the middle line
    if len(sorted_contours) > 10:
        middle_contour = sorted_contours[len(sorted_contours) // 2]  # Contours are 0-indexed
        x, _, w, _ = cv2.boundingRect(middle_contour)
        return x, w
    else:
        return image.shape[1] // 2, 50

def intersection_area(rect1, rect2):
    """
    Calculate the intersection area of two rectangles.

    Parameters:
    - rect1: Tuple containing (x, y, width, height) of the first rectangle.
    - rect2: Tuple containing (x, y, width, height) of the second rectangle.

    Returns:
    - Area of the intersection.
    """
    x_overlap = max(0, min(rect1[0] + rect1[2], rect2[0] + rect2[2]) - max(rect1[0], rect2[0]))
    y_overlap = max(0, min(rect1[1] + rect1[3], rect2[1] + rect2[3]) - max(rect1[1], rect2[1]))
    return x_overlap * y_overlap

def get_col(xml_loghi, image_path):
    """
    Extract column coordinates from the image.

    Parameters:
    - xml_loghi: Path to the XML file containing loghi data.
    - image_path: Path to the image file.

    Returns:
    - List of column coordinates as contours.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found at the specified path")

    # Find text lines and vertical lines
    textlines, leftmost, rightmost = find_loghi_textlines(xml_loghi)
    fg_image, foreground = foreground_extractor(image.copy())
    vertical_lines = vertical_line_finder(image.copy(), foreground, (leftmost, rightmost))
    columns = cv2.subtract(fg_image, vertical_lines)

    # Find contours of the columns
    column_coords, _ = cv2.findContours(columns, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return column_coords

def get_col_row(xml_loghi, image_path):
    """
    Extract column and row coordinates from the image.

    Parameters:
    - xml_loghi: Path to the XML file containing loghi data.
    - image_path: Path to the image file.

    Returns:
    - Tuple containing lists of column and row coordinates as contours.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found at the specified path")

    # Find text lines and vertical lines
    textlines, leftmost, rightmost = find_loghi_textlines(xml_loghi)
    fg_image, foreground = foreground_extractor(image.copy())
    vertical_lines = vertical_line_finder(image.copy(), foreground, (leftmost, rightmost))
    columns = cv2.subtract(fg_image, vertical_lines)

    # Find words and middle line
    loghi_words_dict = find_loghi_words(xml_loghi)
    x, w = find_middle(vertical_lines)
    left_row_boxes, right_row_boxes = get_row_boxes(loghi_words_dict, (x, w))

    # Find contours of the columns
    column_coords, _ = cv2.findContours(columns, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    row_coords = get_row_coords(image.copy(), left_row_boxes, right_row_boxes, (x, w), (leftmost, rightmost))

    return column_coords, row_coords

def get_cells(xml_loghi, image_path):
    """
    Extract table cells by finding intersections between columns and rows.

    Parameters:
    - xml_loghi: Path to the XML file containing loghi data.
    - image_path: Path to the image file.

    Returns:
    - List of table cells as polygons.
    """
    column_coords, row_coords = get_col_row(xml_loghi, image_path)

    # Convert the tuple of numpy arrays to a list of numpy arrays
    column_coords_list = list(column_coords)
    row_coords_list = list(row_coords)

    # Create lists to store the Polygon objects
    column_polygons = []
    row_polygons = []

    # Convert each numpy array in the list to a Polygon object
    for coords in column_coords_list:
        coords_reshaped = np.reshape(coords, (-1, 2))
        coords_tuples = [tuple(coord) for coord in coords_reshaped]
        column_polygons.append(Polygon(coords_tuples))

    for coords in row_coords_list:
        coords_reshaped = np.reshape(coords, (-1, 2))
        coords_tuples = [tuple(coord) for coord in coords_reshaped]
        row_polygons.append(Polygon(coords_tuples))

    # Find intersections between row and column polygons
    table_cells = []
    for row in row_polygons:
        for column in column_polygons:
            if row.intersects(column):
                cell = row.intersection(column)
                if cell.geom_type == 'Polygon':
                    table_cells.append(cell)
                elif cell.geom_type == 'GeometryCollection':
                    # If it's a GeometryCollection, add each polygon
                    polygons = [geom for geom in cell.geoms if geom.geom_type == 'Polygon']
                    table_cells.extend(polygons)

    return table_cells

def shapely_to_opencv_polygon(polygon):
    """
    Convert a Shapely polygon to a NumPy array suitable for OpenCV.

    Parameters:
    - polygon: Shapely polygon.

    Returns:
    - NumPy array of polygon coordinates.
    """
    exterior_coords = np.array(polygon.exterior.coords.xy).T.astype(np.int32)
    return exterior_coords

def draw_polygons_on_image(image, polygons, color=(0, 255, 0), thickness=2):
    """
    Draw polygons on an image using OpenCV.

    Parameters:
    - image: Image on which to draw the polygons.
    - polygons: List of polygons to draw.
    - color: Color of the polygons (default is green).
    - thickness: Thickness of the polygon lines (default is 2).

    Returns:
    - Image with drawn polygons.
    """
    for polygon in polygons:
        polygon_pts = shapely_to_opencv_polygon(polygon)
        cv2.polylines(image, [polygon_pts], isClosed=True, color=color, thickness=thickness)

# Example Usage:
# xml_loghi = '/home/roderickmajoor/Desktop/Master/Thesis/loghi/data/55/page/WBMA00007000010.xml'
# image_path = '/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/WBMA00007000010.jpg'

# cells = get_cells(xml_loghi, image_path)

# image = cv2.imread(image_path)

# draw_polygons_on_image(image, cells)

# Display or save the image with the drawn polygons
# cv2.namedWindow('Cells', cv2.WINDOW_NORMAL)
# cv2.imshow('Cells', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

