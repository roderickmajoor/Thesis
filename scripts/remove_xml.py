import os

def remove_xml_files(root_dir):
    # Walk through each directory and subdirectory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if 'page' folder exists in the current directory
        if 'page' in dirnames:
            page_dir = os.path.join(dirpath, 'page')
            # Iterate through files in the 'page' directory
            for filename in os.listdir(page_dir):
                if filename.endswith(".xml"):
                    file_path = os.path.join(page_dir, filename)
                    # Remove the XML file
                    os.remove(file_path)

# Example usage:
root_directory = "/home/roderickmajoor/Desktop/Master/Thesis/loghi/data"
remove_xml_files(root_directory)
