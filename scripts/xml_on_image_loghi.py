import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt

path = '/home/roderickmajoor/Desktop/Master/Thesis/loghi/data/55/page/WBMA00007000010.xml'

# Load the XML file
tree = ET.parse(path)
root = tree.getroot()

# Define namespaces
ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

# Get image filename, width, and height
image_filename = root.find('page:Page', ns).attrib['imageFilename']
image_width = int(root.find('page:Page', ns).attrib['imageWidth'])
image_height = int(root.find('page:Page', ns).attrib['imageHeight'])

# Load the image
image = cv2.imread('/home/roderickmajoor/Desktop/Master/Thesis/loghi/data/55/' + image_filename)

# Function to draw colored lines
def draw_colored_lines(coordinates, color):
    for i in range(len(coordinates)):
        cv2.line(image, coordinates[i], coordinates[(i+1) % len(coordinates)], color, 2)

# Function to add text to the image
def add_text_to_image(text_content, text_position):
    cv2.putText(image, text_content, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Iterate through TextRegion elements
for text_region in root.findall('.//page:TextRegion', ns):
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

        # Iterate through Word elements
        for word in text_line.findall('page:Word', ns):
            # Get Word coordinates
            word_coords = word.find('page:Coords', ns).attrib.get('points', '')
            word_coordinates = [tuple(map(int, point.split(','))) for point in word_coords.split()]
            # Draw Word coords with grey line
            draw_colored_lines(word_coordinates, (128, 128, 128))

            # Get text content
            text_content = word.find('page:TextEquiv/page:PlainText', ns).text
            # Get text position (average of Word coordinates)
            text_position = (sum(x[0] for x in word_coordinates) // len(word_coordinates), sum(x[1] for x in word_coordinates) // len(word_coordinates))
            # Add text to the image
            add_text_to_image(text_content, text_position)

# Display the image
#cv2.namedWindow('Loghi XML on Image', cv2.WINDOW_NORMAL)
#cv2.imshow('Loghi XML on Image', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
# Display the image with GT layout in the notebook
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
