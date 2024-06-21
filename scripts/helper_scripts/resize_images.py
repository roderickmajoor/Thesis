import cv2
import os

# Path to the folder containing subfolders with jpg files
folder_path = "/media/roderickmajoor/TREKSTOR/Train2/1045/images_split/"

# Function to resize images in a folder
def resize_images(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg"):
                filepath = os.path.join(root, file)
                try:
                    # Read the image using OpenCV
                    img = cv2.imread(filepath)
                    # Get the dimensions of the image
                    height, width = img.shape[:2]
                    # Calculate new dimensions
                    new_width = int(width / 2)
                    new_height = int(height / 2)
                    # Resize the image
                    img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    # Save the resized image, overwrite the original
                    cv2.imwrite(filepath, img_resized)
                    print(f"Resized {file}")
                except Exception as e:
                    print(f"Error resizing {file}: {e}")

# Resize images
resize_images(folder_path)
