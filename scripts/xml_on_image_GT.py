import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt

# Path to the ground truth (GT) XML file
gt_path = '/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/page/WBMA00007000010.xml'

# Load the GT XML file
gt_tree = ET.parse(gt_path)
gt_root = gt_tree.getroot()

# Define namespaces
ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

# Load the image corresponding to the GT
image_filename = gt_root.find('.//page:Page', ns).attrib['imageFilename']
image = cv2.imread('/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/' + image_filename)

# Function to draw colored lines
def draw_colored_lines(coordinates, color):
    for i in range(len(coordinates)):
        cv2.line(image, coordinates[i], coordinates[(i+1) % len(coordinates)], color, 2)

# Function to add text to the image
def add_text_to_image(text_content, text_position):
    cv2.putText(image, text_content, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Iterate through TextRegion elements
for text_region in gt_root.findall('.//page:TextRegion', ns):
    # Get TextRegion coordinates
    text_region_coords = text_region.find('page:Coords', ns).attrib.get('points', '')
    text_region_coordinates = [tuple(map(int, point.split(','))) for point in text_region_coords.split()]
    # Draw TextRegion coords with green line
    draw_colored_lines(text_region_coordinates, (0, 255, 0))

    # Iterate through TextLine elements
    for text_line in text_region.findall('page:TextLine', ns):
        # Get TextLine coordinates
        text_line_coords = text_line.find('page:Coords', ns).attrib.get('points', '')
        text_line_coordinates = [tuple(map(int, point.split(','))) for point in text_line_coords.split()]
        # Draw TextLine coords with blue line
        draw_colored_lines(text_line_coordinates, (255, 0, 0))

        # Get Baseline coordinates
        baseline_coords = text_line.find('page:Baseline', ns).attrib.get('points', '')
        baseline_coordinates = [tuple(map(int, point.split(','))) for point in baseline_coords.split()]
        # Draw Baseline with red line
        draw_colored_lines(baseline_coordinates, (0, 0, 255))

        # Get text content
        text_content = text_line.find('page:TextEquiv/page:Unicode', ns).text
        # Get text position (average of Word coordinates)
        text_position = (sum(x[0] for x in text_line_coordinates) // len(text_line_coordinates), sum(x[1] for x in text_line_coordinates) // len(text_line_coordinates))
        # Add text to the image
        add_text_to_image(text_content, text_position)

# Iterate through TableRegion elements in the GT
for table_region in gt_root.findall('.//page:TableRegion', ns):
    # Get TableRegion coordinates
    table_coords = table_region.find('page:Coords', ns).attrib.get('points', '')
    table_coordinates = [tuple(map(int, point.split(','))) for point in table_coords.split()]
    # Draw TableRegion coords with green line
    draw_colored_lines(table_coordinates, (0, 255, 0))

    # Iterate through TableCell elements
    for table_cell in table_region.findall('page:TableCell', ns):
        # Get TableCell coordinates
        cell_coords = table_cell.find('page:Coords', ns).attrib.get('points', '')
        cell_coordinates = [tuple(map(int, point.split(','))) for point in cell_coords.split()]
        # Draw TableCell coords with blue line
        draw_colored_lines(cell_coordinates, (255, 0, 0))

        # Iterate through TextLine elements
        for text_line in table_cell.findall('page:TextLine', ns):
            # Get TextLine coordinates
            text_line_coords = text_line.find('page:Coords', ns).attrib.get('points', '')
            text_line_coordinates = [tuple(map(int, point.split(','))) for point in text_line_coords.split()]
            # Draw TextLine coords with blue line
            draw_colored_lines(text_line_coordinates, (255, 0, 0))

            # Get Baseline coordinates
            baseline_coords = text_line.find('page:Baseline', ns).attrib.get('points', '')
            baseline_coordinates = [tuple(map(int, point.split(','))) for point in baseline_coords.split()]
            # Draw Baseline with red line
            draw_colored_lines(baseline_coordinates, (0, 0, 255))

            # Get text content
            text_content = text_line.find('page:TextEquiv/page:Unicode', ns).text
            # Get text position (average of Word coordinates)
            text_position = (sum(x[0] for x in text_line_coordinates) // len(text_line_coordinates), sum(x[1] for x in text_line_coordinates) // len(text_line_coordinates))
            # Add text to the image
            add_text_to_image(text_content, text_position)

# Display the image with GT layout
#cv2.namedWindow('Ground Truth Layout', cv2.WINDOW_NORMAL)
#cv2.imshow('Ground Truth Layout', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
