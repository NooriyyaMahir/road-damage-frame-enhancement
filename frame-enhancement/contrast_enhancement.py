import cv2
import os

in_dir = "03_denoised"
out_dir = "04_contrast"
os.makedirs(out_dir, exist_ok=True)

for img_name in os.listdir(in_dir):
    img = cv2.imread(f"{in_dir}/{img_name}", 0)
    enhanced = cv2.equalizeHist(img)
    cv2.imwrite(f"{out_dir}/{img_name}", enhanced)

print("Contrast enhancement completed")

