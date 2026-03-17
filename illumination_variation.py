import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------------
# Member 2 : Illumination Analysis
# CLAHE Contrast Enhancement
# -----------------------------------

# Image Path
image_path = r"01_original_frames\frame_100.png"

if not os.path.exists(image_path):
    print("Error: Image file not found.")
    exit()

# Load grayscale image
gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if gray is None:
    print("Error loading image.")
    exit()

print("Image loaded successfully.")

# -----------------------------------
# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
# -----------------------------------

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_image = clahe.apply(gray)

# -----------------------------------
# Save Output
# -----------------------------------

cv2.imwrite("clahe_output.jpg", clahe_image)
print("CLAHE enhanced image saved.")

# -----------------------------------
# Display Image Comparison
# -----------------------------------

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(gray, cmap='gray')
plt.title("Original Grayscale Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(clahe_image, cmap='gray')
plt.title("CLAHE Enhanced Image")
plt.axis('off')

plt.tight_layout()
plt.show()

# -----------------------------------
# Histogram Comparison
# -----------------------------------

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.hist(gray.ravel(), bins=256, range=[0,256])
plt.title("Original Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(1,2,2)
plt.hist(clahe_image.ravel(), bins=256, range=[0,256])
plt.title("CLAHE Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
