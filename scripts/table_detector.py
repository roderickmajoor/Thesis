import cv2
import numpy as np

def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Binarize image using Otsu's method
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove small connected components (noise)
    cleaned = cv2.bitwise_not(cv2.bitwise_and(binary, cv2.bitwise_not(cv2.erode(binary, None))))

    return cleaned

def find_table_lines(image):
    # Detect horizontal and vertical lines using Hough transform
    lines = cv2.HoughLinesP(image, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Draw lines on image

            # Additional filtering based on line length, rotation, and average blackness
            # You can implement this filtering based on your specific requirements

    return image

# Load image
image = cv2.imread("/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/WBMA00007000010.jpg")

# Preprocess image
processed_image = preprocess_image(image.copy())

# Find and draw table lines
lines_image = find_table_lines(processed_image.copy())

# Display results in resizable windows
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.imshow('Original Image', image)

cv2.namedWindow('Processed Image', cv2.WINDOW_NORMAL)
cv2.imshow('Processed Image', processed_image)

cv2.namedWindow('Lines Image', cv2.WINDOW_NORMAL)
cv2.imshow('Lines Image', lines_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
