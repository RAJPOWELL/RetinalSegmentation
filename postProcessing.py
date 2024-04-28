import cv2
import numpy as np

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(image, table)

def main():
    # Load the image
    img = cv2.imread('files/output/op5.png')

    # Display the original image
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)  # Wait for any key to be pressed

    # Apply gamma correction
    gamma = 50.0  # Set the gamma value (higher gamma for maximum effect)
    img_gamma_corrected = adjust_gamma(img, gamma)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_gamma_corrected, cv2.COLOR_BGR2GRAY)

    # Apply contrast reduction by multiplying by a fraction (e.g., 0.5)
    img_low_contrast = (img_gray * 0.9).astype(np.uint8)

    # Display the processed image
    cv2.imshow('Processed Image', img_low_contrast)
    cv2.waitKey(0)  # Wait for any key to be pressed

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
