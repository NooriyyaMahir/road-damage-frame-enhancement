import cv2
import pywt
import numpy as np
import os
from scipy.stats import skew, kurtosis

# Folder containing images
folder_path = "01_original_frames"   # change this to your folder name

# Get all files in folder
image_files = os.listdir(folder_path)

for file in image_files:

    image_path = os.path.join(folder_path, file)

    image = cv2.imread(image_path)

    if image is None:
        print("Skipping file:", file)
        continue

    print("\nProcessing Image:", file)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Wavelet Transform
    LL, (LH, HL, HH) = pywt.dwt2(gray_image, 'haar')

    # Flatten high-frequency coefficients
    W = HH.flatten()

    # Noise estimation
    sigma = np.median(np.abs(W)) / 0.6745

    # Statistical properties
    mean_val = np.mean(W)
    skewness_val = skew(W)
    kurtosis_val = kurtosis(W)

    print("Estimated Noise Sigma:", sigma)
    print("Mean:", mean_val)
    print("Skewness:", skewness_val)
    print("Kurtosis:", kurtosis_val)

    # Detect Gaussian noise
    if abs(skewness_val) < 0.5 and abs(kurtosis_val - 3) < 1:
        print("Gaussian Noise Detected")
    else:
        print("Noise is not Gaussian")
