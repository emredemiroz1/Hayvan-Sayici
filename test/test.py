import cv2
import numpy as np
import os

# Define the image path
image_path = r'C:\Users\emred\OneDrive\Masaüstü\tartara sayıcı\tartara.jpg'

# Check if the file exists
if not os.path.exists(image_path):
    print(f"Dosya bulunamadı: {image_path}")
else:
    print("Dosya mevcut!")

    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print("Görüntü yüklenemedi.")
    else:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Use adaptive thresholding to segment the cockroaches
        _, threshold_image = cv2.threshold(blurred_image, 100, 255, cv2.THRESH_BINARY_INV)

        # Find contours (blobs)
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image for visualization (optional)
        output_image = image.copy()
        cv2.drawContours(output_image, contours, -1, (0, 255, 0), 1)

        # Count the number of cockroaches (i.e., contours)
        cockroach_count = len(contours)

        # Display the results
        print(f'Total number of cockroaches detected: {cockroach_count}')

        # Optionally, display the processed images
        cv2.imshow('Original Image', image)
        cv2.imshow('Thresholded Image', threshold_image)
        cv2.imshow('Cockroach Detection', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
