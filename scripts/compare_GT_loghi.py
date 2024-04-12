import os
import xml.etree.ElementTree as ET
from nltk.metrics import edit_distance

def calculate_edit_distance(gt_text, htr_output):
    # Calculate the edit distance between GT text and HTR output
    return edit_distance(gt_text, htr_output)

def calculate_word_error_rate(gt_text, htr_output):
    # Compute Word Error Rate (WER)
    wer = calculate_edit_distance(gt_text, htr_output) / len(gt_text)
    return wer

def calculate_character_error_rate(gt_text, htr_output):
    # Compute Character Error Rate (CER)
    cer = calculate_edit_distance(''.join(gt_text), ''.join(htr_output)) / len(''.join(gt_text))
    return cer

def calculate_word_accuracy(gt_text, htr_output):
    # Calculate Word Accuracy
    correct_words = sum(1 for gt, htr in zip(gt_text, htr_output) if gt == htr)
    accuracy = correct_words / len(gt_text)
    return accuracy

def count_matching_words(list1, list2):
    # Convert lists to dictionaries to count occurrences
    dict1 = {}
    dict2 = {}

    # Count occurrences in list 1
    for word in list1:
        dict1[word] = dict1.get(word, 0) + 1

    # Count occurrences in list 2
    for word in list2:
        dict2[word] = dict2.get(word, 0) + 1

    # Find the intersection of the two dictionaries
    matching_words = set(dict1.keys()).intersection(set(dict2.keys()))

    # Count the occurrences based on the minimum count in both lists
    total_count = sum(min(dict1.get(word, 0), dict2.get(word, 0)) for word in matching_words)

    return total_count

def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    return intersection / union if union != 0 else 0

def get_all_xml_files(root_directory):
    xml_files = []
    for root, _, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".xml"):
                xml_files.append(os.path.join(root, file))
    return xml_files

def extract_text_from_xml_gt(xml_file):
    ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract text content from all TextEquiv elements
    text_content = []
    for text_equiv in root.findall('.//page:TextEquiv', ns):
        text = text_equiv.find('page:Unicode', ns).text
        if text:
            text_content.append(text)

    return text_content

def extract_text_from_xml_loghi(xml_file):
    ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract text content from all TextEquiv elements
    text_content = []
    for word in root.findall('.//page:Word', ns):
        text_equiv = word.find('page:TextEquiv', ns)
        if text_equiv is not None:
            text = text_equiv.find('page:Unicode', ns).text
            if text:
                text_content.append(text)

    return text_content

def compute_mean_metrics(all_xml_GT, all_xml_loghi):
    edit_distance_total = 0
    wer_total = 0
    cer_total = 0
    matching_count_total = 0
    jaccard_similarity_total = 0
    num_images = 0

    for data_gt, data_loghi in zip(all_xml_GT, all_xml_loghi):
        all_text_gt = extract_text_from_xml_gt(data_gt)
        all_text_loghi = extract_text_from_xml_loghi(data_loghi)
        all_text_loghi = [item for item in all_text_loghi if any(char.isdigit() for char in item)]

        edit_distance_total += edit_distance(all_text_gt, all_text_loghi)
        wer_total += calculate_word_error_rate(all_text_gt, all_text_loghi)
        cer_total += calculate_character_error_rate(all_text_gt, all_text_loghi)
        matching_count_total += count_matching_words(all_text_gt, all_text_loghi) / len(all_text_loghi)
        jaccard_similarity_total += jaccard_similarity(all_text_gt, all_text_loghi)
        num_images += 1

    mean_edit_distance = edit_distance_total / num_images
    mean_wer = wer_total / num_images
    mean_cer = cer_total / num_images
    mean_matching_count = matching_count_total / num_images
    mean_jaccard_similarity = jaccard_similarity_total / num_images

    return mean_edit_distance, mean_wer, mean_cer, mean_matching_count, mean_jaccard_similarity

def main():
    all_xml_GT = get_all_xml_files('/home/roderickmajoor/Desktop/Master/Thesis/GT_data/')
    all_xml_loghi = get_all_xml_files('/home/roderickmajoor/Desktop/Master/Thesis/loghi/data/')

    list1_filenames = [os.path.basename(path) for path in all_xml_GT]
    list2_filenames = [os.path.basename(path) for path in all_xml_loghi]
    if list1_filenames == list2_filenames:
        print("The lists are in the same order based on filenames.")

    mean_edit_distance, mean_wer, mean_cer, mean_matching_count, mean_jaccard_similarity = compute_mean_metrics(all_xml_GT, all_xml_loghi)

    print("Mean Edit Distance:", mean_edit_distance)
    print("Mean Word Error Rate:", mean_wer)
    print("Mean Character Error Rate:", mean_cer)
    print("Mean Matching Count:", mean_matching_count)
    print("Mean Jaccard Similarity:", mean_jaccard_similarity)

main()