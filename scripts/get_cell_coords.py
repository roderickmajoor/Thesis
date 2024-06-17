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
    # Create a black mask of the same size as the image
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #mask = np.zeros_like(image)

    # Iterate over each word and draw its bounding box on the mask
    for word_data in loghi_words_dict.values():
        coords = word_data['coords']
        points = parse_coords(coords)
        # Convert coordinates to numpy array and reshape for drawing
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        # Draw filled polygon on the mask
        cv2.fillPoly(image, [pts], (255, 255, 255))

    # Invert the mask
    #mask = cv2.bitwise_not(mask)

    # Apply the mask to the image
    #result = cv2.bitwise_and(image, mask)

    return image

def dilate_horizontally(image):
    """
    Perform dilation on an image with a horizontal kernel.

    Parameters:
    - image: Image

    Returns:
    - Dilated image.
    """

    # Get image dimensions
    height, width = image.shape[:2]

    # Split the image into left and right halves
    left_half = image[:, :width // 2]
    right_half = image[:, width // 2:]

    # Create horizontal kernels for left and right sides
    left_kernel = np.ones((1, 3), np.uint8)
    right_kernel = np.ones((1, 3), np.uint8)
    #left_kernel = np.array([[1, 0, 0]], dtype=np.uint8)

    #right_kernel = np.array([[0, 0, 1]], dtype=np.uint8)

    # Perform dilation on the left half with the horizontal kernel pointing left
    dilated_left = cv2.dilate(left_half, left_kernel, iterations=200)

    # Perform dilation on the right half with the horizontal kernel pointing right
    dilated_right = cv2.dilate(right_half, right_kernel, iterations=200)

    # Perform dilation on the left half with the horizontal kernel pointing left
    dilated_left = cv2.erode(dilated_left, left_kernel, iterations=5)

    # Perform dilation on the right half with the horizontal kernel pointing right
    dilated_right = cv2.erode(dilated_right, right_kernel, iterations=5)

    # Combine the dilated left and right halves
    dilated_image = np.hstack((dilated_left, dilated_right))

    return dilated_image


def get_row_boxes(loghi_words_dict, middle):
    boxes = []
    x_middle, w_middle = middle

    for word_data in loghi_words_dict.values():
        coords = word_data['coords']
        points = parse_coords(coords)
        # Convert coordinates to numpy array
        pts = np.array(points, np.int32)
        polygon = Polygon(pts)
        boxes.append(polygon)

    #mean_height = np.mean([box[3] for box in boxes])
    #threshold = mean_height

    # Separate boxes into two groups: left and right
    left_boxes = [box for box in boxes if max(box.exterior.coords, key=lambda x: x[0])[0] < x_middle]
    right_boxes = [box for box in boxes if min(box.exterior.coords, key=lambda x: x[0])[0] >= x_middle+w_middle]

    # Function to get row boxes from a list of boxes
    def get_rows(boxes):
        # Sort boxes by x-coordinate
        boxes.sort(key=lambda box: box.centroid.x)

        # Group boxes with similar y-coordinate together
        rows = [] # A list[] containing lists[] of bounding boxes() of words
        row_y_values = [] # A list[] containing tuples (y_min, y_max) for each row
        for box in boxes:
            min_y_box = min(box.exterior.coords, key=lambda x: x[1])[1]
            max_y_box = max(box.exterior.coords, key=lambda x: x[1])[1]
            #min_prev = min(prev_box.exterior.coords, key=lambda x: x[1])[1]
            #max_prev = max(prev_box.exterior.coords, key=lambda x: x[1])[1]

            if row_y_values:
                curr_best = np.inf
                best_row = len(row_y_values)
                for i, y_values in enumerate(row_y_values):
                    if (max_y_box > y_values[0] and max_y_box < y_values[1]) or (y_values[1] > min_y_box and y_values[0] < max_y_box):
                        if abs(max_y_box-y_values[1]) + abs(min_y_box-y_values[0]) < curr_best:
                            best_row = i
                            curr_best = abs(max_y_box-y_values[1]) + abs(min_y_box-y_values[0])
                        #rows[i].append(box.buffer(0))
                        #row_y_values[i] = (min_y_box, max_y_box)
                        #break
                    #elif i == len(row_y_values)-1:
                    #    rows.append([box.buffer(0)])
                    #    row_y_values.append((min_y_box, max_y_box))
                if best_row == len(row_y_values):
                    rows.append([box.buffer(0)])
                    row_y_values.append((min_y_box, max_y_box))
                elif best_row != len(row_y_values):
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
                # Append the result to the list
                row_boxes.append(combined_polygon)

        return row_boxes

    # Get row boxes for left and right groups
    left_row_boxes = get_rows(left_boxes)
    right_row_boxes = get_rows(right_boxes)

    return left_row_boxes, right_row_boxes

# Function to convert shapely multipolygon to OpenCV contours
def multipolygon_to_contours(multipolygon):
    contours = []
    for polygon in multipolygon:
        contour = np.array(polygon.exterior.coords, dtype=np.int32)
        contours.append(contour)
    return contours

def extend_polygon_to_edges(polygon, left_edge, right_edge):
    extended_polygon = []
    for vertex in polygon:
        #print(vertex)
        # Get x-coordinate
        x_coord = vertex[0]
        # Extend to the left edge (x=0)
        left_extended_vertex = [min(x_coord, left_edge), vertex[1]]
        # Extend to the right edge (x=100)
        right_extended_vertex = [max(x_coord, right_edge), vertex[1]]
        extended_polygon.append(left_extended_vertex)
        extended_polygon.append(right_extended_vertex)

    return np.array(extended_polygon)

def extend_polygon_to_edges1(polygon, left_edge, right_edge):
    # Find the top left, top right, bottom left, and bottom right points
    top_left = min(polygon, key=lambda coord: coord[0] + coord[1])
    bottom_right = max(polygon, key=lambda coord: coord[0] + coord[1])
    top_right = min(polygon, key=lambda coord: coord[0] - coord[1])
    bottom_left = max(polygon, key=lambda coord: coord[0] - coord[1])

    # Extend the top left and bottom left vertices to the left edge
    top_left_extended = [min(top_left[0], left_edge), top_left[1]]
    bottom_left_extended = [min(bottom_left[0], left_edge), bottom_left[1]]

    # Extend the top right and bottom right vertices to the right edge
    top_right_extended = [max(top_right[0], right_edge), top_right[1]]
    bottom_right_extended = [max(bottom_right[0], right_edge), bottom_right[1]]

    extended_polygon = polygon.copy()

    values_to_append = np.array([top_left_extended, top_right_extended, bottom_left_extended, bottom_right_extended])

    # Append values to the existing array
    new_array = np.append(extended_polygon, values_to_append, axis=0)


    return new_array



def get_row_coords(image, left_boxes, right_boxes, middle, edges):
    x_middle, w_middle = middle
    leftmost, rightmost = edges
    row_coords = []
    all_boxes = left_boxes + right_boxes

    for box in all_boxes:
        if isinstance(box, MultiPolygon):
            # Convert multipolygon to OpenCV contours
            contours = multipolygon_to_contours(box.geoms)
            # Draw original multipolygon
            #cv2.polylines(mask, contours, isClosed=True, color=(255, 255, 255), thickness=2)

            combined_contour = np.concatenate(contours)

            #print(combined_contour)

            if box in left_boxes:
                extended_contour = extend_polygon_to_edges(combined_contour, left_edge=leftmost, right_edge=x_middle)
            else:
                extended_contour = extend_polygon_to_edges(combined_contour, left_edge=x_middle+w_middle, right_edge=rightmost)

            # Approximate contours to convex hulls
            extended_contour = cv2.convexHull(extended_contour)


        # Else Polygon
        else:
            # Convert the polygon coordinates to a NumPy array
            points = np.array(box.exterior.coords, dtype=np.int32)

            if box in left_boxes:
                extended_contour = extend_polygon_to_edges(points, left_edge=leftmost, right_edge=x_middle)
            else:
                extended_contour = extend_polygon_to_edges(points, left_edge=x_middle+w_middle, right_edge=rightmost)

            extended_contour = cv2.convexHull(extended_contour)

        # Shift the entire contour upwards by 20 pixels
        #shift_amount_x = 0  # No shift in the x-direction
        #shift_amount_y = -30  # Shift by 20 pixels upwards

        # Add the shift amounts to all y coordinates of the convex hull points
        #shifted_convex_hull = extended_contour.copy()
        #shifted_convex_hull[:, 0, 1] += shift_amount_y

        row_coords.append(extended_contour)

    return row_coords

def draw_rows(image, left_boxes, right_boxes, middle):
    x_middle, w_middle = middle
    # Create a black mask of the same size as the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(image)

    all_boxes = left_boxes + right_boxes

    for box in all_boxes:
        if isinstance(box, MultiPolygon):
            # Convert multipolygon to OpenCV contours
            contours = multipolygon_to_contours(box.geoms)
            # Draw original multipolygon
            #cv2.polylines(mask, contours, isClosed=True, color=(255, 255, 255), thickness=2)

            combined_contour = np.concatenate(contours)

            #print(combined_contour)

            if box in left_boxes:
                extended_contour = extend_polygon_to_edges(combined_contour, left_edge=0, right_edge=x_middle)
            else:
                extended_contour = extend_polygon_to_edges(combined_contour, left_edge=x_middle+w_middle, right_edge=image.shape[1])

            # Approximate contours to convex hulls
            extended_contour = cv2.convexHull(extended_contour)


        # Else Polygon
        else:
            # Convert the polygon coordinates to a NumPy array
            points = np.array(box.exterior.coords, dtype=np.int32)

            if box in left_boxes:
                extended_contour = extend_polygon_to_edges(points, left_edge=0, right_edge=x_middle)
            else:
                extended_contour = extend_polygon_to_edges(points, left_edge=x_middle+w_middle, right_edge=image.shape[1])

            extended_contour = cv2.convexHull(extended_contour)

        cv2.polylines(mask, [extended_contour], isClosed=True, color=(255, 255, 255), thickness=2)


    return mask

def find_middle(image):
    # Find contours in the binary image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on x-coordinate
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # Get the middle line
    if len(sorted_contours) > 10:
        middle_contour = sorted_contours[len(sorted_contours)//2]  # Contours are 0-indexed
        # Get the bounding box coordinates of the sixth contour
        x, _, w, _ = cv2.boundingRect(middle_contour)

        return x, w
    else:
        return image.shape[1]//2, 50

def intersection_area(rect1, rect2):
    x_overlap = max(0, min(rect1[0] + rect1[2], rect2[0] + rect2[2]) - max(rect1[0], rect2[0]))
    y_overlap = max(0, min(rect1[1] + rect1[3], rect2[1] + rect2[3]) - max(rect1[1], rect2[1]))
    return x_overlap * y_overlap

def create_table(column_coords, row_coords, loghi_words_dict):
    sorted_columns = sorted(column_coords, key=lambda c: cv2.boundingRect(c)[0])
    sorted_rows = sorted(row_coords, key=lambda c: cv2.boundingRect(c)[1])
    sorted_loghi_words_dict = sorted(loghi_words_dict.values(), key=lambda x: min(coord[0] for coord in x['coords']))

    matrix = [[[] for _ in range(len(sorted_columns))] for _ in range(len(sorted_rows))]

    for word_data in sorted_loghi_words_dict:
        coords = word_data['coords']
        points = parse_coords(coords)
        word_box = cv2.boundingRect(np.array(points, np.int32))

        # Calculate overlap with columns and rows
        column_overlap = [intersection_area(word_box, cv2.boundingRect(col)) for col in sorted_columns]
        row_overlap = [intersection_area(word_box, cv2.boundingRect(row)) for row in sorted_rows]

        # Find column and row with maximum overlap
        max_column_index = np.argmax(column_overlap)
        max_row_index = np.argmax(row_overlap)

        matrix[max_row_index][max_column_index].append(word_data['text'])

    df = pd.DataFrame(matrix, columns=range(len(matrix[0])), index=range(len(matrix)))

    # Define a function to remove rows with empty lists
    def remove_empty_rows(table):
        return table[table.applymap(lambda x: isinstance(x, list) and len(x) > 0).any(axis=1)]

    # Split the DataFrame into two separate tables
    if df.shape[1] > 8:
        table1 = df.iloc[:, :df.shape[1]//2]  # First half columns
        table2 = df.iloc[:, df.shape[1]//2:]  # Last half columns

        # Remove empty rows from each table
        table1 = remove_empty_rows(table1)
        table2 = remove_empty_rows(table2)

        return [table1, table2]
    else:
        df = remove_empty_rows(df)
        return df

def get_col(xml_loghi, image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found at the specified path")

    textlines, leftmost, rightmost = find_loghi_textlines(xml_loghi)
    fg_image, foreground = foreground_extractor(image.copy())
    vertical_lines = vertical_line_finder(image.copy(), foreground, (leftmost, rightmost))
    columns = cv2.subtract(fg_image, vertical_lines)

    column_coords, _ = cv2.findContours(columns, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return column_coords

def get_col_row(xml_loghi, image_path):
    #xml_loghi = '/home/roderickmajoor/Desktop/Master/Thesis/loghi/data/55/page/WBMA00007000010.xml'
    #image_path = '/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/WBMA00007000010.jpg'

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found at the specified path")

    textlines, leftmost, rightmost = find_loghi_textlines(xml_loghi)
    fg_image, foreground = foreground_extractor(image.copy())
    vertical_lines = vertical_line_finder(image.copy(), foreground, (leftmost, rightmost))
    columns = cv2.subtract(fg_image, vertical_lines)

    loghi_words_dict = find_loghi_words(xml_loghi)
    x, w = find_middle(vertical_lines)
    left_row_boxes, right_row_boxes = get_row_boxes(loghi_words_dict, (x,w))
    #rows_image = draw_rows(image.copy(), left_row_boxes, right_row_boxes, (x,w))

    column_coords, _ = cv2.findContours(columns, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    row_coords = get_row_coords(image.copy(), left_row_boxes, right_row_boxes, (x,w), (leftmost, rightmost))
    #df = create_table(column_coords, row_coords, loghi_words_dict)

    #result_image = cv2.add(vertical_lines, rows_image)


    # Invert the binary mask
    #binary_mask = cv2.bitwise_not(result_image)
    # Convert the binary mask to 3 channels (to match the original color image)
    #binary_mask = cv2.merge([binary_mask, binary_mask, binary_mask])
    #result_image = cv2.bitwise_and(image, binary_mask)

    #cv2.namedWindow('Rows', cv2.WINDOW_NORMAL)
    #cv2.imshow('Rows', rows_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return column_coords, row_coords

def get_cells(xml_loghi, image_path):
    #xml_loghi = '/home/roderickmajoor/Desktop/Master/Thesis/loghi/data/55/page/WBMA00007000010.xml'
    #image_path = '/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/WBMA00007000010.jpg'

    column_coords, row_coords = get_col_row(xml_loghi, image_path)

    # Convert the tuple of numpy arrays to a list of numpy arrays
    column_coords_list = list(column_coords)
    row_coords_list = list(row_coords)

    # Create a list to store the Polygon objects
    column_polygons = []
    row_polygons = []

    # Convert each numpy array in the list to a Polygon object
    for coords in column_coords_list:
        # Reshape the array to (N, 2)
        coords_reshaped = np.reshape(coords, (-1, 2))

        # Convert the reshaped array to a list of tuples
        coords_tuples = [tuple(coord) for coord in coords_reshaped]

        # Create a Polygon object and add it to the list
        column_polygons.append(Polygon(coords_tuples))

    for coords in row_coords_list:
        # Reshape the array to (N, 2)
        coords_reshaped = np.reshape(coords, (-1, 2))

        # Convert the reshaped array to a list of tuples
        coords_tuples = [tuple(coord) for coord in coords_reshaped]

        # Create a Polygon object and add it to the list
        row_polygons.append(Polygon(coords_tuples))

    # Now you can use these polygon lists in the intersection code
    table_cells = []

    for row in row_polygons:
        for column in column_polygons:
            # Check if the row and column intersect
            if row.intersects(column):
                # Get the intersection area (i.e., a cell)
                cell = row.intersection(column)
                #table_cells.append(cell)
                if cell.geom_type == 'Polygon':
                    table_cells.append(cell)
                elif cell.geom_type == 'GeometryCollection':
                    # If it's a GeometryCollection, iterate over its geometries and add each polygon
                    polygons = [geom for geom in cell.geoms if geom.geom_type == 'Polygon']
                    table_cells.extend(polygons)

    #image = cv2.imread(image_path)

    #draw_polygons_on_image(image, table_cells)

    # Display or save the image with the drawn polygons
    #cv2.namedWindow('Cells', cv2.WINDOW_NORMAL)
    #cv2.imshow('Cells', image)
    #cv2.imwrite('/home/roderickmajoor/Desktop/Master/Thesis/images/found_cells.jpg', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return table_cells

def shapely_to_opencv_polygon(polygon):
    """Converts a Shapely polygon to a NumPy array suitable for OpenCV."""
    exterior_coords = np.array(polygon.exterior.coords.xy).T.astype(np.int32)
    return exterior_coords

def draw_polygons_on_image(image, polygons, color=(0, 255, 0), thickness=2):
    """Draws polygons on an image using OpenCV."""
    for polygon in polygons:
        polygon_pts = shapely_to_opencv_polygon(polygon)
        cv2.polylines(image, [polygon_pts], isClosed=True, color=color, thickness=thickness)

#xml_loghi = '/home/roderickmajoor/Desktop/Master/Thesis/loghi/data/55/page/WBMA00007000010.xml'
#image_path = '/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/WBMA00007000010.jpg'

#cells = get_cells(xml_loghi, image_path)

#image = cv2.imread(image_path)

#draw_polygons_on_image(image, cells)

# Display or save the image with the drawn polygons
#cv2.namedWindow('Cells', cv2.WINDOW_NORMAL)
#cv2.imshow('Cells', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


