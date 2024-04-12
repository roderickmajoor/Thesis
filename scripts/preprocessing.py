import cv2
import numpy as np
import os

def remove_noise(image):
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 3, 21)
    return denoised_image

def convert_to_grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def normalize(image):
    normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalized_image

def blur(image):
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    return blurred

def adaptive_threshold(image):
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
    return binary

def binarize(image):
    _, binarized_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binarized_image

def correct_skew(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    angle = rect[2]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image

def thinning(image):
    skeleton = np.zeros(image.shape, dtype=np.uint8)
    skeleton = cv2.ximgproc.thinning(image, skeleton, cv2.ximgproc.THINNING_ZHANGSUEN)
    return skeleton

def normalize_position_size(image):
    x, y, w, h = cv2.boundingRect(image)
    image = image[y:y+h, x:x+w]
    image = cv2.resize(image, (32, 32), interpolation = cv2.INTER_AREA)
    return image

def normalize_strength(image):
    # Add your implementation for strength normalization here
    return image

def morphological(image):
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    detected_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255,255,255), 2)

    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
    result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

    return result

# Load the image
image_path = '/home/roderickmajoor/Desktop/Master/Thesis/loghi/data/55/WBMA00007000010.jpg'
image = cv2.imread(image_path)

# Perform preprocessing steps
image = remove_noise(image)
#image = convert_to_grayscale(image)
#image = binarize(image)
image = blur(image)
#image = adaptive_threshold(image)
#image = morphological(image)


# Create a resizable window
#cv2.namedWindow('Preprocessed Image', cv2.WINDOW_NORMAL)

# Display the image in the resizable window
#cv2.imshow('Preprocessed Image', image)

# Wait for a key press and close the window
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cv2.imwrite('/home/roderickmajoor/Desktop/Master/Thesis/loghi/data_preprocessed/55/WBMA00007000010_noise_blur_2.jpg', image)
# Create a directory to save the processed images
"""
output_directory = '/home/roderickmajoor/Desktop/Master/Thesis/loghi/data_preprocessed/55'
os.makedirs(output_directory, exist_ok=True)

# Define the order of preprocessing steps
preprocessing_steps = [remove_noise, convert_to_grayscale, binarize]

# Iterate through all combinations of preprocessing steps
for i in range(1, len(preprocessing_steps) + 1):
    for j in range(len(preprocessing_steps) - i + 1):
        # Apply the preprocessing steps for the current combination
        processed_image = image.copy()
        steps_to_apply = preprocessing_steps[j:j+i]

        # If binarize is in the steps_to_apply and grayscale_image is not defined
        if binarize in steps_to_apply and convert_to_grayscale not in steps_to_apply:
            continue

        for step in steps_to_apply:
            processed_image = step(processed_image)

        # Generate a filename based on the applied preprocessing steps
        filename = '_'.join([step.__name__ for step in steps_to_apply]) + '.jpg'
        output_path = os.path.join(output_directory, filename)

        # Save the processed image
        cv2.imwrite(output_path, processed_image)
        print(f"Saved {filename}")
"""