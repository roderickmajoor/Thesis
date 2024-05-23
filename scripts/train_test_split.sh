#!/bin/bash

# Define source and destination directories
source_dir="/home/roderickmajoor/Desktop/Master/Thesis/GT_data"
train_dir="/home/roderickmajoor/Desktop/Master/Thesis/Train_Test/Train"
test_dir="/home/roderickmajoor/Desktop/Master/Thesis/Train_Test/Test"

# Create train and test directories
mkdir -p "$train_dir" "$train_dir/page" "$test_dir" "$test_dir/page"

# Select 56 random images and copy them to the train directory
find "$source_dir" -type f -name '*.jpg' | shuf -n 56 | while read -r image; do
    cp "$image" "$train_dir"
    xml_file="${image%.jpg}.xml"
    subdir=$(dirname "$(dirname "$image")")
    cp "$source_dir/$subdir/page/$(basename "$xml_file")" "$train_dir/page"
done

# Copy the remaining images to the test directory
find "$source_dir" -type f -name '*.jpg' | while read -r image; do
    if ! grep -q "$(basename "$image")" <<< "$(ls "$train_dir")"; then
        cp "$image" "$test_dir"
        xml_file="${image%.jpg}.xml"
        subdir=$(dirname "$(dirname "$image")")
        cp "$source_dir/$subdir/page/$(basename "$xml_file")" "$test_dir/page"
    fi
done

