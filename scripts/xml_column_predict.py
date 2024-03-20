import xml.etree.ElementTree as ET
import cv2
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans

path = '/home/roderickmajoor/Desktop/Master/Thesis/loghi/data/page/WBMB00008000060.xml'

# Load the XML file
tree = ET.parse(path)
root = tree.getroot()

# Define namespaces
ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

# Get Image Filename
image_filename = root.find('page:Page', ns).attrib['imageFilename']

min_area_threshold = 3000

# Extract the coordinates of all TextLine elements
all_textline_coordinates = []
for text_region in root.findall('.//page:TextRegion', ns):
    for textline in text_region.findall('.//page:TextLine', ns):
        coords_elem = textline.find('page:Coords', ns)
        if coords_elem is not None:
            points = coords_elem.attrib.get('points', '')
            coordinates = [tuple(map(int, point.split(','))) for point in points.split()]
            leftmost_x = min(coord[0] for coord in coordinates)
            bounding_box_area = (max(coord[0] for coord in coordinates) - leftmost_x) * (max(coord[1] for coord in coordinates) - min(coord[1] for coord in coordinates))
            if bounding_box_area >= min_area_threshold:
                all_textline_coordinates.append(leftmost_x)

# Reshape the data to a 2D array for clustering
X = np.array(all_textline_coordinates).reshape(-1, 1)

# Initialize the KMeans clustering model with 6 clusters
kmeans = KMeans(n_clusters=6)

# Fit the model to the data
kmeans.fit(X)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Get the cluster centers
cluster_centers = kmeans.cluster_centers_

# Reshape and transform the cluster centers to integers
cluster_centers_1d = cluster_centers.flatten().astype(int)

# Load the image
image = cv2.imread('/home/roderickmajoor/Desktop/Master/Thesis/loghi/data/' + image_filename)

# Draw vertical lines for column borders on the image
for border in cluster_centers_1d:
    cv2.line(image, (border,0), (border, image.shape[0]), (0, 255, 0), 2)  # Green color for the lines

# Display the image with vertical lines for column borders
cv2.namedWindow('Image with Column Borders', cv2.WINDOW_NORMAL)
cv2.imshow('Image with Column Borders', image)
cv2.waitKey(0)
cv2.destroyAllWindows()