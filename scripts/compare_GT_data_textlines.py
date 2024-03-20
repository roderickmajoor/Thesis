import xml.etree.ElementTree as ET
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

# Parse the ground truth XML file
tree_loghi = ET.parse('/home/roderickmajoor/Desktop/Master/Thesis/loghi/data/55/page/WBMA00007000010.xml')
root_loghi = tree_loghi.getroot()

# Parse the original XML file containing words
tree_gt = ET.parse('/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/page/WBMA00007000010.xml')
root_gt = tree_gt.getroot()

# Function to convert string coordinates to Shapely Polygon
def string_to_polygon(coords_str):
    # Split the string by spaces to get individual coordinate pairs
    coord_pairs = coords_str.split()

    # Convert each coordinate pair to tuple of floats
    points = [tuple(map(float, pair.split(','))) for pair in coord_pairs]

    # Create a Shapely Polygon object from the points
    polygon = Polygon(points)

    return polygon

# Function to calculate the intersection over union (IoU) between two polygon shapes
def calculate_iou(poly1, poly2):
    # Create Shapely Polygon objects from the polygon coordinates
    poly1_shapely = make_valid(string_to_polygon(poly1))
    poly2_shapely = make_valid(string_to_polygon(poly2))

    # Calculate intersection area
    intersection_area = poly1_shapely.intersection(poly2_shapely).area

    # Calculate union area
    union_area = poly1_shapely.union(poly2_shapely).area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def main():

  table_cells_gt = root_gt.findall('.//page:TableCell', ns)
  text_regions_gt = root_gt.findall('.//page:TextRegion', ns)
  all_regions_gt = text_regions_gt + table_cells_gt

  words_loghi = root_loghi.findall('.//page:Word', ns)

  num_gt = len(all_regions_gt)
  num_loghi = len(words_loghi)
  iou_matrix = np.zeros((num_gt, num_loghi))

  # Initialize dictionaries to store region information
  gt_regions_dict = {}
  loghi_words_dict = {}

  # Compare words in original data with table cells in ground truth
  for i, region_gt in enumerate(all_regions_gt):
      coords_gt = region_gt.find('page:Coords', ns).attrib['points']
      text_gt = region_gt.find('.//page:TextEquiv/page:Unicode', ns).text if region_gt.find('.//page:TextEquiv/page:Unicode', ns) is not None else ""
      gt_regions_dict[i] = {'coords': coords_gt, 'text': text_gt}

      for j, word_loghi in enumerate(words_loghi):
          coords_loghi = word_loghi.find('page:Coords', ns).attrib['points']
          text_loghi = word_loghi.find('page:TextEquiv/page:Unicode', ns).text
          loghi_words_dict[j] = {'coords': coords_loghi, 'text': text_loghi}

          iou_matrix[i, j] = calculate_iou(coords_gt, coords_loghi)

  # Find the maximum IoU value for each GT region
  max_iou_indices = np.argmax(iou_matrix, axis=1)

  # Keep track of matched regions to ensure only one match per region
  matched_gt_regions = set()
  matched_loghi_regions = set()

  # Iterate over the GT regions and find the corresponding loghi word with the highest IoU
  matches = {}
  for gt_index, loghi_index in enumerate(max_iou_indices):
      if loghi_index not in matched_loghi_regions:  # Ensure loghi region is not already matched
          iou_value = iou_matrix[gt_index, loghi_index]
          if gt_index not in matched_gt_regions:  # Ensure GT region is not already matched
              matches[gt_index] = loghi_index
              matched_gt_regions.add(gt_index)
              matched_loghi_regions.add(loghi_index)

  data_gt = []
  data_loghi = []

  # Print the matched regions
  for gt_index, loghi_index in matches.items():
      iou_value = iou_matrix[gt_index, loghi_index]
      text_gt = gt_regions_dict[gt_index]['text']
      text_loghi = loghi_words_dict[loghi_index]['text']
      data_gt.append(text_gt)
      data_loghi.append(text_loghi)
      #print(f"GT region {gt_index} matched with loghi word {loghi_index} with IoU {iou_value}: GT text - {text_gt}, Loghi text - {text_loghi}")

  indices_to_remove = [i for i, value in enumerate(data_gt) if value == '' or value is None]
  # Reverse the list of indices so that removing elements doesn't affect indices of subsequent elements
  indices_to_remove.reverse()
  for i in indices_to_remove:
    del data_gt[i]
    del data_loghi[i]

  return data_gt, data_loghi