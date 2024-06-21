import xml.etree.ElementTree as ET
import cv2
import numpy as np

# Parse the provided XML string
xml_columns = '/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/page/WBMA00007000010_columns_found.xml'
tree = ET.parse(xml_columns)
root = tree.getroot()

# Load the image
image_path = '/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/WBMA00007000010.jpg'
image = cv2.imread(image_path)

# Define a function to draw regions on the image
def draw_regions(image, regions):
    for region in regions:
        coords = region.find('Coords').attrib['points']
        points = [(int(p.split(',')[0]), int(p.split(',')[1])) for p in coords.split()]
        cv2.polylines(image, [np.array(points)], True, (0, 255, 0), thickness=2)

# Get the Page element
page_element = root.find('.//Page')

# Get all TextRegion elements under the Page
text_regions = page_element.findall('.//TextRegion')

# Draw the regions on the image
draw_regions(image, text_regions)

# Display the image with the regions
cv2.namedWindow('Image with Regions', cv2.WINDOW_NORMAL)
cv2.imshow('Image with Regions', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
