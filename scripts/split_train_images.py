import os
import shutil

# Path to the folder containing the images
images_folder = '/media/roderickmajoor/TREKSTOR/Train/images'

# Create a directory to store the split parts
output_folder = '/media/roderickmajoor/TREKSTOR/Train/images_parts'
os.makedirs(output_folder, exist_ok=True)

# Define the maximum number of images per folder
max_images_per_folder = 5

# Iterate over the images in the folder
images = os.listdir(images_folder)
num_images = len(images)

# Calculate the number of parts needed
num_parts = (num_images + max_images_per_folder - 1) // max_images_per_folder

# Split the images into parts
for i in range(num_parts):
    # Create a subfolder for the part
    part_folder = os.path.join(output_folder, f'part{i+1}')
    os.makedirs(part_folder, exist_ok=True)

    # Create a 'page' folder inside the part folder
    page_folder = os.path.join(part_folder, 'page')
    os.makedirs(page_folder, exist_ok=True)

    # Move images to the part folder
    start_idx = i * max_images_per_folder
    end_idx = min((i + 1) * max_images_per_folder, num_images)
    for j in range(start_idx, end_idx):
        image = images[j]
        shutil.move(os.path.join(images_folder, image), part_folder)

print("Images have been split into parts successfully.")
