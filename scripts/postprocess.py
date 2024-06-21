# This script implements the function to post-process the found loghi words and
# textlines that are used in other scripts.

import xml.etree.ElementTree as ET
import numpy as np

ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

def find_loghi_words(xml):
    """
    Extracts words from a PAGE XML file, performs substitutions on characters,
    and returns a dictionary of word coordinates and cleaned text.

    Args:
    - xml (str): Path to the PAGE XML file.

    Returns:
    - dict: A dictionary where each key is a word index and each value is another dictionary with 'coords' and 'text' keys.
    """
    loghi_words_dict = {}
    tree_loghi = ET.parse(xml)
    root_loghi = tree_loghi.getroot()
    words_loghi = root_loghi.findall('.//page:Word', ns)

    # Frequently occurring substitutions made by HTR system where number -> char
    replacements = {
        'a': '1', 'n': '7', 's': '8', 'r': '2', 'i': '1', 't': '1', 'o': '0', 'g': '9',
        'e': '1', 'd': '1', 'ƒ': '1', 'p': '7', 'S': '8', 'k': '3', 'R': '7', 'u': '1',
        'v': '1', 'l': '1', 'h': '2', 'f': '1', 'B': '3', 'b': '6', 'm': '9', 'w': '1',
        'y': '9', 'c': '1', 'C': '1', 'I': '1', 'N': '1', 'E': '6', 'z': '3', 'q': '9'
    }

    for j, word_loghi in enumerate(words_loghi):
        coords_loghi = word_loghi.find('page:Coords', ns).attrib['points']
        text_loghi = word_loghi.find('page:TextEquiv/page:Unicode', ns).text

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

    return loghi_words_dict

def find_loghi_textlines(xml_loghi):
    """
    Extracts textlines from a PAGE XML file, calculates the leftmost and rightmost points of textlines.

    Args:
    - xml_loghi (str): Path to the PAGE XML file.

    Returns:
    - list: A list of textline coordinates.
    - float: The leftmost x-coordinate among all textlines.
    - float: The rightmost x-coordinate among all textlines.
    """
    # Load the XML data
    tree = ET.parse(xml_loghi)
    root = tree.getroot()

    # Namespace
    namespace = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    # List to store textlines
    textlines = []

    # Retrieve textline coords points and calculate leftmost and rightmost points
    leftmost_point = float('inf')
    rightmost_point = float('-inf')

    for text_region in root.findall('.//ns:TextRegion', namespace):
        for text_line in text_region.findall('.//ns:TextLine', namespace):
            coords = text_line.find('ns:Coords', namespace).attrib['points']
            textlines.append(coords)
            for coord in coords.split(' '):
                x, _ = map(int, coord.split(','))
                if x < leftmost_point:
                    leftmost_point = x
                if x > rightmost_point:
                    rightmost_point = x

    return textlines, leftmost_point, rightmost_point
