import os
import shutil
import random

# Define your directories
source_dir = "/home/roderickmajoor/Desktop/Master/Thesis/GT_data"
train_dir = "/home/roderickmajoor/Desktop/Master/Thesis/Train_Test/Train"
test_dir = "/home/roderickmajoor/Desktop/Master/Thesis/Train_Test/Test"

# Get list of all subdirectories
subdirs = [os.path.join(source_dir, d) for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

# Get list of all jpg files in all subdirs
jpg_files = []
for subdir in subdirs:
    jpg_files.extend([os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith('.jpg')])

# Randomly select 56 jpg files for the training set
train_files = random.sample(jpg_files, 56)

# The rest of the files will be used for the test set
test_files = list(set(jpg_files) - set(train_files))

# Copy training jpg and xml files to the train directory
for file in train_files:
    shutil.copy(file, train_dir)
    xml_file = os.path.join(os.path.dirname(file), 'page', os.path.basename(file).replace('.jpg', '.xml'))
    if os.path.exists(xml_file):
        shutil.copy(xml_file, os.path.join(train_dir, 'page'))

# Copy testing jpg and xml files to the test directory
for file in test_files:
    shutil.copy(file, test_dir)
    xml_file = os.path.join(os.path.dirname(file), 'page', os.path.basename(file).replace('.jpg', '.xml'))
    if os.path.exists(xml_file):
        shutil.copy(xml_file, os.path.join(test_dir, 'page'))
