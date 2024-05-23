#!/bin/bash

# Set the source and destination directories
source_dir="/media/roderickmajoor/TREKSTOR/Train"
dest_dir="$source_dir/images"

# Ensure the destination directory exists
mkdir -p "$dest_dir"

# Convert each PDF in the source directory to images
for pdf_file in "$source_dir"/*.pdf; do
    # Get the base filename without extension
    base_name=$(basename -- "$pdf_file" .pdf)

    # Convert the PDF to images using pdftoppm
    pdftoppm -jpeg "$pdf_file" "$dest_dir/$base_name"
done
