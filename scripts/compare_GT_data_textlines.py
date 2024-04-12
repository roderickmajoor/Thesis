import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from collections import Counter
from shapely.geometry import Polygon
from shapely.validation import make_valid

ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

# Parse loghi file
tree_loghi = ET.parse('/home/roderickmajoor/Desktop/Master/Thesis/loghi/data/55/page/WBMA00007000010.xml')
#tree_loghi = ET.parse('/home/roderickmajoor/Desktop/Master/Thesis/loghi/data_preprocessed/55/page/WBMA00007000010_noise_blur_2.xml')
root_loghi = tree_loghi.getroot()

# Parse GT File
tree_gt = ET.parse('/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/page/WBMA00007000010.xml')
root_gt = tree_gt.getroot()

# Load the image corresponding to the GT
image_filename = root_gt.find('.//page:Page', ns).attrib['imageFilename']
image = cv2.imread('/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/' + image_filename)

def get_all_xml_files(root_directory):
    xml_files_by_subdir = {}
    for root, subdirs, files in os.walk(root_directory):
        xml_files = [os.path.join(root, file) for file in files if file.endswith(".xml")]
        if xml_files:
            xml_files_by_subdir[root] = xml_files
    return xml_files_by_subdir

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

# Function to draw colored lines on image
def draw_colored_lines(coordinates, color):
    for i in range(len(coordinates)):
        cv2.line(image, coordinates[i], coordinates[(i+1) % len(coordinates)], color, 2)

# Function to draw the xml on the image
def xml_on_image(matched_data):
    for dicts in matched_data:
        coords = dicts['coords_gt']
        text_line_coordinates = [tuple(map(int, point.split(','))) for point in coords.split()]

        if dicts['matched'] == True:
            # Draw TextLine coords with green line
            draw_colored_lines(text_line_coordinates, (0, 255, 0))
        else:
            # Draw TextLine coords with red line
            draw_colored_lines(text_line_coordinates, (0, 0, 255))

    cv2.imwrite('/home/roderickmajoor/Desktop/Master/Thesis/images/matched_errors.jpg', image)

# Function to calculate edit distance between two strings
def edit_distance_operations(s1, s2):
    insertions = 0
    deletions = 0
    substitutions = 0
    matches = 0

    chars_inserted = []
    chars_deleted = []
    chars_subsituted = []

    m = len(s1) + 1
    n = len(s2) + 1

    # Initialize the table
    tbl = [[(0, []) for _ in range(n)] for _ in range(m)]
    for i in range(1, m): tbl[i][0] = (i, [('D', s1[i-1], '')] * i)
    for j in range(1, n): tbl[0][j] = (j, [('I', '', s2[j-1])] * j)

    # Fill the table
    for i in range(1, m):
        for j in range(1, n):
            del_cost = tbl[i-1][j][0] + 1
            ins_cost = tbl[i][j-1][0] + 1
            sub_cost = tbl[i-1][j-1][0] + (s1[i-1] != s2[j-1])
            if del_cost < ins_cost and del_cost < sub_cost:
                tbl[i][j] = (del_cost, tbl[i-1][j][1] + [('D', s1[i-1], '')])
            elif ins_cost < del_cost and ins_cost < sub_cost:
                tbl[i][j] = (ins_cost, tbl[i][j-1][1] + [('I', '', s2[j-1])])
            else:
                tbl[i][j] = (sub_cost, tbl[i-1][j-1][1] + [('S' if s1[i-1] != s2[j-1] else 'M', s1[i-1], s2[j-1])])

    for operations in tbl[-1][-1][1]:
        if operations[0] == 'I':
            insertions += 1
            chars_inserted.append(operations[2])
        elif operations[0] == 'D':
            deletions += 1
            chars_deleted.append(operations[1])
        elif operations[0] == 'S':
            substitutions += 1
            chars_subsituted.append((operations[1], operations[2]))
        else:
            matches += 1


    return insertions, chars_inserted, deletions, chars_deleted, substitutions, chars_subsituted, matches

# Function to plot the character frequency for insertions, deletions and substitutions
def plot_frequency(insertions, deletions, substitutions, tot_insertions, tot_deletions, tot_substitutions, tot_matches):
    # Count the frequency of each character in insertions and deletions
    insertion_counts = Counter(insertions)
    deletion_counts = Counter(deletions)

    # Count the frequency of each tuple in substitutions
    substitution_counts = Counter(substitutions)

    # Plot the frequency of insertions
    plt.figure(figsize=(12, 6))
    plt.bar(insertion_counts.keys(), insertion_counts.values())
    plt.title('Frequency of Insertions')
    plt.xlabel('Character')
    plt.ylabel('Frequency')
    plt.show()

    # Plot the frequency of deletions
    plt.figure(figsize=(12, 6))
    plt.bar(deletion_counts.keys(), deletion_counts.values())
    plt.title('Frequency of Deletions')
    plt.xlabel('Character')
    plt.ylabel('Frequency')
    plt.show()

    # Plot the frequency of substitutions
    plt.figure(figsize=(12, 6))
    plt.bar([f'{k[0]} -> {k[1]}' for k in substitution_counts.keys()], substitution_counts.values())
    plt.title('Frequency of Substitutions')
    plt.xlabel('Substitution (m -> n)')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.show()

    # Plot the frequency of deletions
    plt.figure(figsize=(12, 6))
    plt.bar(['Insertions', 'Deletions', 'Substitutions', 'Matches'], [tot_insertions, tot_deletions, tot_substitutions, tot_matches])
    plt.title('Frequency of Operations')
    plt.xlabel('Operations')
    plt.ylabel('Frequency')
    plt.show()

def plot_all_data(data_subdirs):
    # Create bar plot for insertions for every subdir
    subdirs = list(data_subdirs.keys())
    insertions = [data_subdirs[subdir]['subdir_insertions'] for subdir in subdirs]
    plt.bar(subdirs, insertions)
    plt.xlabel('Subdirectory')
    plt.ylabel('Insertions')
    plt.title('Insertions for Every Subdirectory')
    plt.show()

    # Create bar plot for deletions for every subdir
    deletions = [data_subdirs[subdir]['subdir_deletions'] for subdir in subdirs]
    plt.bar(subdirs, deletions)
    plt.xlabel('Subdirectory')
    plt.ylabel('Deletions')
    plt.title('Deletions for Every Subdirectory')
    plt.show()

    # Create bar plot for substitutions for every subdir
    substitutions = [data_subdirs[subdir]['subdir_substitutions'] for subdir in subdirs]
    plt.bar(subdirs, substitutions)
    plt.xlabel('Subdirectory')
    plt.ylabel('Substitutions')
    plt.title('Substitutions for Every Subdirectory')
    plt.show()

    # Create bar plot for matches for every subdir
    matches = [data_subdirs[subdir]['subdir_matches'] for subdir in subdirs]
    plt.bar(subdirs, matches)
    plt.xlabel('Subdirectory')
    plt.ylabel('Matches')
    plt.title('Matches for Every Subdirectory')
    plt.show()

    # Total insertions, deletions, substitutions, and matches of all subdirs combined
    total_insertions = sum(insertions)
    total_deletions = sum(deletions)
    total_substitutions = sum(substitutions)
    total_matches = sum(matches)

    # Create bar plot for total insertions, deletions, substitutions, and matches
    categories = ['Insertions', 'Deletions', 'Substitutions', 'Matches']
    totals = [total_insertions, total_deletions, total_substitutions, total_matches]
    plt.bar(categories, totals)
    plt.xlabel('Categories')
    plt.ylabel('Total')
    plt.title('Total Frequency of Categories for all Subdirectories Combined')
    plt.show()

    # Total frequency of each char inserted
    all_chars_inserted = [char for subdir in data_subdirs.values() for char in subdir['subdir_chars_inserted']]
    char_inserted_counts = Counter(all_chars_inserted)
    plt.bar(char_inserted_counts.keys(), char_inserted_counts.values())
    plt.xlabel('Character Inserted')
    plt.ylabel('Frequency')
    plt.title('Total Frequency of Each Character Inserted')
    plt.show()

    # Total frequency of each char deleted
    all_chars_deleted = [char for subdir in data_subdirs.values() for char in subdir['subdir_chars_deleted']]
    char_deleted_counts = Counter(all_chars_deleted)
    plt.bar(char_deleted_counts.keys(), char_deleted_counts.values())
    plt.xlabel('Character Deleted')
    plt.ylabel('Frequency')
    plt.title('Total Frequency of Each Character Deleted')
    plt.show()

    # Total frequency of each char substituted
    all_chars_substituted = [(original, substitution) for subdir in data_subdirs.values() for original, substitution in subdir['subdir_chars_substituted']]
    char_substituted_counts = Counter(all_chars_substituted)
    labels = [f'{k[0]} -> {k[1]}' for k in char_substituted_counts.keys()]
    values = char_substituted_counts.values()
    plt.bar(labels, values)
    # Only show labels where frequency > 1
    for i, v in enumerate(values):
        if v <= 50:
            labels[i] = ''
    plt.xticks(labels)
    plt.xlabel('Character Substituted')
    plt.ylabel('Frequency')
    plt.title('Total Frequency of Each Character Substituted (Original -> Substitution)')
    plt.show()

def one_image_data(root_gt, root_loghi):

    table_cells_gt = root_gt.findall('.//page:TableCell', ns)
    text_regions_gt = root_gt.findall('.//page:TextRegion', ns)
    all_regions_gt = text_regions_gt + table_cells_gt

    words_loghi = root_loghi.findall('.//page:Word', ns)

    tot_insertions = 0
    tot_deletions = 0
    tot_substitutions = 0
    tot_matches = 0

    all_chars_inserted = []
    all_chars_deleted = []
    all_chars_substitued = []

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

        for j, word_loghi in enumerate(words_loghi):
            coords_loghi = word_loghi.find('page:Coords', ns).attrib['points']
            text_loghi = word_loghi.find('page:TextEquiv/page:Unicode', ns).text
            loghi_words_dict[j] = {'coords': coords_loghi, 'text': text_loghi}

            iou = calculate_iou(coords_gt, coords_loghi)
            if iou > best_iou:
                best_iou = iou
                matched_word_index = j

        if best_iou > 0.0:
            gt_text = gt_regions_dict[i]['text']
            htr_text = loghi_words_dict[matched_word_index]['text']
            gt_coords = gt_regions_dict[i]['coords']
            htr_coords = loghi_words_dict[matched_word_index]['coords']

            insertions, chars_inserted, deletions, chars_deleted, substitutions, chars_subsituted, matches = edit_distance_operations(gt_text, htr_text)

            tot_insertions += insertions
            tot_deletions += deletions
            tot_substitutions += substitutions
            tot_matches += matches

            all_chars_inserted.extend(chars_inserted)
            all_chars_deleted.extend(chars_deleted)
            all_chars_substitued.extend(chars_subsituted)

            matched = False if insertions + deletions + substitutions > 0 else True

            matched_data.append({
                'text_gt': gt_text,
                'coords_gt': gt_coords,
                'text_htr': htr_text,
                'coords_htr': htr_coords,
                'matched': matched
            })

    return matched_data, tot_insertions, tot_deletions, tot_substitutions, tot_matches, all_chars_inserted, all_chars_deleted, all_chars_substitued

def all_image_data():
    all_xml_GT = get_all_xml_files('/home/roderickmajoor/Desktop/Master/Thesis/GT_data/')
    all_xml_loghi = get_all_xml_files('/home/roderickmajoor/Desktop/Master/Thesis/loghi/data/')

    data_subdirs = {}

    # Loop over subdirs
    for (subdir_GT, xml_files_GT), (subdir_loghi, xml_files_loghi) in zip(all_xml_GT.items(), all_xml_loghi.items()):
        last_dir_name = os.path.normpath(subdir_GT).split(os.sep)[-2]

        subdir_total_chars = 0
        subdir_insertions = 0
        subdir_deletions = 0
        subdir_substitutions = 0
        subdir_matches = 0

        subdir_chars_inserted = []
        subdir_chars_deleted = []
        subdir_chars_substitued = []

        # Loop over XML files for GT and loghi directories simultaneously
        for xml_file_GT, xml_file_loghi in zip(xml_files_GT, xml_files_loghi):
            # Parse loghi file
            tree_loghi = ET.parse(xml_file_loghi)
            root_loghi = tree_loghi.getroot()

            # Parse GT File
            tree_gt = ET.parse(xml_file_GT)
            root_gt = tree_gt.getroot()

            matched_data, file_insertions, file_deletions, file_substitutions, file_matches, file_chars_inserted, file_chars_deleted, file_chars_substitued = one_image_data(root_gt, root_loghi)

            length = sum([len(d["text_gt"]) for d in matched_data])
            subdir_total_chars += length
            subdir_insertions += file_insertions
            subdir_deletions += file_deletions
            subdir_substitutions += file_substitutions
            subdir_matches += file_matches

            subdir_chars_inserted.extend(file_chars_inserted)
            subdir_chars_deleted.extend(file_chars_deleted)
            subdir_chars_substitued.extend(file_chars_substitued)

        data_subdirs[last_dir_name] = {
            'subdir_total_chars': subdir_total_chars,
            'subdir_insertions': subdir_insertions,
            'subdir_deletions': subdir_deletions,
            'subdir_substitutions': subdir_substitutions,
            'subdir_matches': subdir_matches,
            'subdir_chars_inserted': subdir_chars_inserted,
            'subdir_chars_deleted': subdir_chars_deleted,
            'subdir_chars_substituted': subdir_chars_substitued
        }

    return data_subdirs


#matched_data, tot_insertions, tot_deletions, tot_substitutions, tot_matches, all_chars_inserted, all_chars_deleted, all_chars_substitued = one_image_data()

#print("Total Insertions:" + str(tot_insertions))
#print("Total Deletions:" + str(tot_deletions))
#print("Total Substitutions:" + str(tot_substitutions))
#print("Total Matches:" + str(tot_matches))

#print("All Chars Inserted:" + str(all_chars_inserted))
#print("All Chars Deleted:" + str(all_chars_deleted))
#print("All Chars Substituted:" + str(all_chars_substitued))

#plot_frequency(all_chars_inserted, all_chars_deleted, all_chars_substitued, tot_insertions, tot_deletions, tot_substitutions, tot_matches)
#xml_on_image(matched_data)

###############################

#data_subdirs = all_image_data()
#plot_all_data(data_subdirs)