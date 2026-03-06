import cv2
import os

in_dir = "02_grayscale"
out_dir = "03_denoised"
os.makedirs(out_dir, exist_ok=True)

for img_name in os.listdir(in_dir):
    gray = cv2.imread(f"{in_dir}/{img_name}", 0)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(f"{out_dir}/{img_name}", denoised)

print("Noise removal completed")
