import os
import shutil

def collect_images(source_dir, target_dir):
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Get the list of subdirectories
    subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for subdir in subdirs:
        subdir_path = os.path.join(source_dir, subdir)
        for root, _, files in os.walk(subdir_path):
            for file in files:
                if file.endswith('.jpg'):  # Modify this if you want to include other image formats
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(target_dir, file)

                    # If the destination file already exists, rename the new file to avoid overwriting
                    if os.path.exists(dst_file):
                        base, extension = os.path.splitext(file)
                        count = 1
                        new_dst_file = os.path.join(target_dir, f"{base}_{count}{extension}")
                        while os.path.exists(new_dst_file):
                            count += 1
                            new_dst_file = os.path.join(target_dir, f"{base}_{count}{extension}")
                        dst_file = new_dst_file

                    shutil.copy2(src_file, dst_file)
                    print(f"Copied {src_file} to {dst_file}")

# Define the source and target directories
source_dir = '/media/roderickmajoor/TREKSTOR/Train/images_parts'
target_dir = '/media/roderickmajoor/TREKSTOR/Train/images'

collect_images(source_dir, target_dir)
