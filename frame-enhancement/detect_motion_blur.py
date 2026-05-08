import cv2
import os
import numpy as np

input_folder = "01_original_frames"
output_folder = "frames_enhanced"

threshold = 100  # blur detection threshold

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):

    path = os.path.join(input_folder, file)

    image = cv2.imread(path)

    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur detection
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    print(file, "Sharpness:", laplacian_var)

    if laplacian_var < threshold:
        print("Blur Detected → Applying Sharpening")

        # Sharpening kernel
        kernel = np.array([[0,-1,0],
                           [-1,5,-1],
                           [0,-1,0]])

        sharpened = cv2.filter2D(image, -1, kernel)

        cv2.imwrite(os.path.join(output_folder, file), sharpened)

    else:
        print("Image is Sharp → Saved without change")

        cv2.imwrite(os.path.join(output_folder, file), image)



