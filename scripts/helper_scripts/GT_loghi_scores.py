import xml.etree.ElementTree as ET
import os
import Levenshtein
import jiwer
import glob
from shapely.geometry import Polygon
from shapely.validation import make_valid

ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

def get_all_xml_files(root_directory):
    xml_files_by_subdir = {}
    for root, subdirs, files in os.walk(root_directory):
        xml_files = [os.path.join(root, file) for file in files if file.endswith(".xml")]
        if xml_files:
            xml_files_by_subdir[root] = xml_files
    return xml_files_by_subdir

# Function to retrieve all XML files in a directory and its subdirectories
def retrieve_xml_files(directory):
    xml_files = []
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(directory):
        # Check if 'page' directory exists
        if 'page' in root:
            # Find XML files in the 'page' directory
            xml_files.extend(os.path.join(root, file) for file in files if file.endswith(".xml") and not file.endswith("_columns_found.xml"))
    return xml_files

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

def one_image_data(root_gt, root_loghi):

    table_cells_gt = root_gt.findall('.//page:TableCell', ns)
    text_regions_gt = root_gt.findall('.//page:TextRegion', ns)
    all_regions_gt = text_regions_gt + table_cells_gt

    words_loghi = root_loghi.findall('.//page:Word', ns)

    # Frequently occuring substituions made by htr system where number -> char
    replacements = {
        'a': '1', 'n': '7', 's': '8', 'r': '2', 'i': '1', 't': '1', 'o': '0', 'g': '9',
        'e': '1', 'd': '1', 'ƒ': '1', 'p': '7', 'S': '8', 'k': '3', 'R': '7', 'u': '1',
        'v': '1', 'l': '1', 'h': '2', 'f': '1', 'B': '3', 'b': '6', 'm': '9', 'w': '1',
        'y': '9', 'c': '1', 'C': '1', 'I': '1', 'N': '1', 'E': '6', 'z': '3', 'q': '9'
    }

    # Initialize dictionaries to store region information
    gt_regions_dict = {}
    loghi_words_dict = {}

    matched_data = []

    # Compare words in original data with table cells in ground truth
    for i, region_gt in enumerate(all_regions_gt):
        coords_gt = region_gt.find('page:Coords', ns).attrib['points']
        text_gt = region_gt.find('.//page:TextEquiv/page:Unicode', ns).text if region_gt.find('.//page:TextEquiv/page:Unicode', ns) is not None else ""

        if text_gt is None or text_gt == '':
            continue

        gt_regions_dict[i] = {'coords': coords_gt, 'text': text_gt}

        best_iou = 0.0
        matched_word_index = -1

        """
        for j, word_loghi in enumerate(words_loghi):
            coords_loghi = word_loghi.find('page:Coords', ns).attrib['points']
            text_loghi = word_loghi.find('page:TextEquiv/page:Unicode', ns).text
            loghi_words_dict[j] = {'coords': coords_loghi, 'text': text_loghi}

            iou = calculate_iou(coords_gt, coords_loghi)
            if iou > best_iou:
                best_iou = iou
                matched_word_index = j
        """

        for j, word_loghi in enumerate(words_loghi):
            coords_loghi = word_loghi.find('page:Coords', ns).attrib['points']
            text_loghi = word_loghi.find('page:TextEquiv/page:Unicode', ns).text
            #loghi_words_dict[j] = {'coords': coords_loghi, 'text': text_loghi}

            # Check if text contains letters or numbers
            if any(c.isalnum() for c in text_loghi):
                # Remove characters that are not letters or numbers
                cleaned_text = ''.join(c for c in text_loghi if c.isalnum())

                # Replace specific characters if the text contains both letters and numbers
                if any(c.isdigit() for c in cleaned_text) and any(c.isalpha() for c in cleaned_text) or len(cleaned_text) == 1:
                    for key, value in replacements.items():
                        cleaned_text = cleaned_text.replace(key, value)

                # Check if cleaned text is a single character and not a number
                if len(cleaned_text) > 1 or cleaned_text.isdigit():
                    loghi_words_dict[j] = {'coords': coords_loghi, 'text': cleaned_text}

            #iou = calculate_iou(coords_gt, coords_loghi)
            #if iou > best_iou:
            #    best_iou = iou
            #    matched_word_index = j
                    iou = calculate_iou(coords_gt, coords_loghi)
                    if iou > best_iou:
                        best_iou = iou
                        matched_word_index = j

        if best_iou > 0.0:
            gt_text = gt_regions_dict[i]['text']
            htr_text = loghi_words_dict[matched_word_index]['text']
            gt_coords = gt_regions_dict[i]['coords']
            htr_coords = loghi_words_dict[matched_word_index]['coords']



            matched_data.append({
                'text_gt': gt_text,
                'coords_gt': gt_coords,
                'text_htr': htr_text,
                'coords_htr': htr_coords,
            })

    return matched_data

def all_image_data():
    all_xml_GT = retrieve_xml_files('/home/roderickmajoor/Desktop/Master/Thesis/GT_data/')
    all_xml_loghi = retrieve_xml_files('/home/roderickmajoor/Desktop/Master/Thesis/loghi/data/')
    all_matches = []

    list1_filenames = [os.path.basename(path) for path in all_xml_GT]
    list2_filenames = [os.path.basename(path) for path in all_xml_loghi]
    if list1_filenames == list2_filenames:
        print("The lists are in the same order based on filenames.")

    for xml_gt, xml_loghi in zip(all_xml_GT, all_xml_loghi):

        # Parse Loghi File
        tree_loghi = ET.parse(xml_loghi)
        root_loghi = tree_loghi.getroot()

        # Parse GT File
        tree_gt = ET.parse(xml_gt)
        root_gt = tree_gt.getroot()

        matched_data = one_image_data(root_gt, root_loghi)

        all_matches.extend(matched_data)

    return all_matches

def calculate_metrics(data):
    total_gt_length = 0
    total_htr_length = 0
    total_edit_distance = 0

    for entry in data:
        text_gt = entry['text_gt']
        text_htr = entry['text_htr']

        # Calculate edit distance
        edit_distance = Levenshtein.distance(text_gt, text_htr)
        total_edit_distance += edit_distance

        # Update total lengths
        total_gt_length += len(text_gt)
        total_htr_length += len(text_htr)

    # Calculate Character Error Rate (CER)
    cer = total_edit_distance / total_gt_length

    # Calculate Word Error Rate (WER)
    wer = jiwer.wer(
        [entry['text_gt'] for entry in data],
        [entry['text_htr'] for entry in data]
    )

    return wer, cer, total_edit_distance


all_matches = all_image_data()

wer, cer, edit_distance = calculate_metrics(all_matches)

print("WER:", wer)
print("CER:", cer)
print("Total Edit Distance:", edit_distance)




