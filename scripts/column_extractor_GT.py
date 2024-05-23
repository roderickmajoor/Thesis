import xml.etree.ElementTree as ET
import cv2

ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

# Parse the XML file
tree = ET.parse('/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/page/WBMA00007000010.xml')
image = cv2.imread('/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/WBMA00007000010.jpg')
gt_root = tree.getroot()

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
        # Draw lines between the points
        cv2.line(image, top_left, top_right, (0, 0, 255), 2)
        cv2.line(image, top_left, bottom_left, (0, 0, 255), 2)
        cv2.line(image, top_right, bottom_right, (0, 0, 255), 2)
        cv2.line(image, bottom_left, bottom_right, (0, 0, 255), 2)

# Show the image with column rectangles
cv2.namedWindow('Columns', cv2.WINDOW_NORMAL)
cv2.imshow('Columns', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
