import cv2
import os

from src.noise_removal import remove_noise
from src.contrast_enhancement import enhance_contrast
from src.sharpening import sharpen_image

def enhance_pipeline(input_folder, output_folder):

    files = os.listdir(input_folder)

    for file in files:

        path = os.path.join(input_folder, file)

        image = cv2.imread(path)

        if image is None:
            continue

        # Noise removal
        denoised = remove_noise(image)

        # Contrast enhancement
        contrast = enhance_contrast(denoised)

        # Sharpening
        sharpened = sharpen_image(contrast)

        save_path = os.path.join(output_folder, file)


        cv2.imwrite(save_path, sharpened)
