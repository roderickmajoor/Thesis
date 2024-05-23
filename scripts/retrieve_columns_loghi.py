import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point
from postprocess import find_loghi_words


def get_columns(xml):
    column_tree = ET.parse(xml)
    column_root = column_tree.getroot()

    # Get the Page element
    page_element = column_root.find('.//Page')

    # Get all TextRegion elements under the Page
    text_regions = page_element.findall('.//TextRegion')

    column_bb = []

    for region in text_regions:
        coords = region.find('Coords').attrib['points']
        points = [(int(p.split(',')[0]), int(p.split(',')[1])) for p in coords.split()]
        column_bb.append((points))

    return column_bb

def find_column_for_word(word_coords, columns):
    """
    Find the column index that contains the centroid of the given word.

    Parameters:
    - word_coords: List of tuples representing the coordinates of the word.
    - columns: List of tuples representing the coordinates of the columns.

    Returns:
    - The index of the column that contains the word's centroid, or None if not found.
    """
    # Function to convert string coordinates to Shapely Polygon
    def string_to_polygon(coords_str):
        # Split the string by spaces to get individual coordinate pairs
        coord_pairs = coords_str.split()

        # Convert each coordinate pair to tuple of floats
        points = [tuple(map(float, pair.split(','))) for pair in coord_pairs]

        # Create a Shapely Polygon object from the points
        polygon = Polygon(points)

        return polygon

    word_polygon = string_to_polygon(word_coords)
    for column_id, column_coords in enumerate(columns):
        column_polygon = Polygon(column_coords)
        centroid = word_polygon.centroid
        if column_polygon.contains(centroid):
            return column_id
    return None

def assign_words_to_columns(loghi_words_dict, columns):
    """
    Assign words to columns based on their coordinates.

    Parameters:
    - loghi_words_dict: Dictionary containing word data with coordinates.
    - columns: List of tuples representing the coordinates of the columns.

    Returns:
    - A dictionary where keys are column indices and values are lists of words belonging to each column.
    """
    columns_with_text = {}
    for word_id, word_data in loghi_words_dict.items():
        word_coords = word_data['coords']
        word_text = word_data['text']
        column_id = find_column_for_word(word_coords, columns)
        if column_id is not None:
            if column_id not in columns_with_text:
                columns_with_text[column_id] = {'coords': columns[column_id], 'words': []}
            columns_with_text[column_id]['words'].append(word_text)

    # Sort columns based on the x-coordinate of their leftmost point
    sorted_columns = sorted(columns_with_text.values(), key=lambda x: x['coords'][0][0])

    return sorted_columns

def convert_to_dataframe(columns_with_text):
    """
    Convert the sorted columns with words into a Pandas DataFrame.

    Parameters:
    - columns_with_text: List of dictionaries representing sorted columns with their words.

    Returns:
    - A Pandas DataFrame where each column contains words belonging to each column of the original data.
    """
    # Extract words from sorted columns
    words_data = []
    for column_data in columns_with_text:
        column_words = column_data['words']
        words_data.append(column_words)

    # Determine the maximum number of rows needed
    max_rows = max(len(words) for words in words_data)

    # Pad shorter word lists with empty strings to match the maximum number of rows
    padded_words_data = [words + [''] * (max_rows - len(words)) for words in words_data]

    # Create DataFrame
    df = pd.DataFrame(padded_words_data).transpose()

    return df

def retrieve_columns(xml_columns, xml_loghi):
    columns = get_columns(xml_columns)
    loghi_words_dict = find_loghi_words(xml_loghi)
    result = assign_words_to_columns(loghi_words_dict, columns)
    df = convert_to_dataframe(result)

    return df

#xml_columns = '/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/page/WBMA00007000010_columns_found.xml'
#xml_loghi = '/home/roderickmajoor/Desktop/Master/Thesis/loghi/data/55/page/WBMA00007000010.xml'

#columns = get_columns(xml_columns)
#loghi_words_dict = find_loghi_words(xml_loghi)
#result = assign_words_to_columns(loghi_words_dict, columns)
#df = convert_to_dataframe(result)
#print(df)
