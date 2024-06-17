#!/bin/bash

# Set the source and destination directories
source_dir="/media/roderickmajoor/TREKSTOR/Train2"
dest_dir="$source_dir/images"

# Ensure the destination directory exists
mkdir -p "$dest_dir"

# Desired maximum dimension for width or height
max_dimension=6000

# Convert each PDF in the source directory to images
for pdf_file in "$source_dir"/*.pdf; do
    # Get the base filename without extension
    base_name=$(basename -- "$pdf_file" .pdf)

    # Convert the PDF to images using pdftoppm with scaling
    pdftoppm -jpeg -scale-to "$max_dimension" "$pdf_file" "$dest_dir/$base_name"
done
