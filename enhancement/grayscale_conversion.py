import cv2
import os

in_dir = "01_original_frames"
out_dir = "02_grayscale"
os.makedirs(out_dir, exist_ok=True)

for img_name in os.listdir(in_dir):
    img = cv2.imread(f"{in_dir}/{img_name}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{out_dir}/{img_name}", gray)

print("Grayscale conversion done")
