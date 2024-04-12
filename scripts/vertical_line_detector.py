import cv2
import numpy as np

def detect_vertical_lines(image_path):
    # Read the image
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create a mask where white regions of the thresholded image are non-zero
    mask = cv2.threshold(thresh, 200, 255, cv2.THRESH_BINARY)[1]

    #Apply the mask to the original image
    result = cv2.bitwise_or(image, cv2.merge([mask, mask, mask]))
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result = cv2.GaussianBlur(result, (3,3), 0)

    result = cv2.threshold(result, 190, 255, cv2.THRESH_BINARY)[1]

    # Define the kernel for erosion and dilation
    kernel_erode = np.ones((15, 3), np.uint8)
    kernel_dilate = np.ones((10, 3), np.uint8)

    # Erode the black regions
    result = cv2.dilate(result, kernel_dilate, iterations=1)

    # Dilate the eroded image
    result = cv2.erode(result, kernel_erode, iterations=3)

    # Display the result in a resizable window
    cv2.namedWindow('Vertical Lines Detected', cv2.WINDOW_NORMAL)
    cv2.imshow('Vertical Lines Detected', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = '/home/roderickmajoor/Desktop/Master/Thesis/GT_data/55/WBMA00007000010.jpg'
detect_vertical_lines(image_path)
