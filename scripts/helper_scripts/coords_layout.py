import os
import xml.etree.ElementTree as ET
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid
from shapely import centroid

#Function to convert string coordinates to Shapely Polygon
def string_to_polygon(coords_str):
    # Split the string by spaces to get individual coordinate pairs
    coord_pairs = coords_str.split()

    # Convert each coordinate pair to tuple of floats
    points = [tuple(map(float, pair.split(','))) for pair in coord_pairs]

    # Create a Shapely Polygon object from the points
    polygon = Polygon(points)

    return (centroid(polygon)).y

def extract_text_from_xml_loghi(xml_file):
    ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract text content from all TextEquiv elements
    text_content = []
    for word in root.findall('.//page:Word', ns):
        text_equiv = word.find('page:TextEquiv', ns)
        coords = string_to_polygon(word.find('page:Coords', ns).attrib['points'])
        if text_equiv is not None:
            text = text_equiv.find('page:Unicode', ns).text
            if text:
                text_content.append((text, coords))

    return text_content

def cluster_words(word_coords, num_clusters):
    """
    Cluster words based on their y-coordinates using hierarchical clustering.

    Args:
    - word_coords (list): A list of tuples containing words and their corresponding y-coordinates.
    - num_clusters (int): The number of clusters to generate.

    Returns:
    - word_clusters (list): A list of clusters, where each cluster is a list of word-coordinate tuples.
    """

    # Extract y-coordinates
    y_coords = np.array([coord[1] for coord in word_coords])

    # Compute pairwise distances
    distances = np.abs(y_coords[:, np.newaxis] - y_coords)

    # Perform hierarchical clustering
    clusters = linkage(distances, method='complete')

    # Determine clusters
    cluster_labels = fcluster(clusters, num_clusters, criterion='maxclust')

    # Assign words to clusters
    word_clusters = [[] for _ in range(num_clusters)]
    for i, label in enumerate(cluster_labels):
        word_clusters[label - 1].append(word_coords[i])

    return word_clusters


text_content = extract_text_from_xml_loghi('/home/roderickmajoor/Desktop/Master/Thesis/loghi/data/55/page/WBMA00007000010.xml')
num_clusters = 52


clusters = cluster_words(text_content, num_clusters)
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {cluster}")