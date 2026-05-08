import cv2
import os

in_dir = "04_contrast"
out_dir = "05_sharpened"
os.makedirs(out_dir, exist_ok=True)

for img_name in os.listdir(in_dir):
    img = cv2.imread(f"{in_dir}/{img_name}", 0)
    lap = cv2.Laplacian(img, cv2.CV_64F)
    sharp = cv2.convertScaleAbs(img - lap)
    cv2.imwrite(f"{out_dir}/{img_name}", sharp)

print("Image sharpening completed")

