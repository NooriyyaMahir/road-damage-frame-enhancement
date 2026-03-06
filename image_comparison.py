import cv2
import matplotlib.pyplot as plt

img_name = "frame_200.png"  

orig = cv2.imread(f"01_original_frames/{img_name}")
gray = cv2.imread(f"02_grayscale/{img_name}", 0)
den = cv2.imread(f"03_denoised/{img_name}", 0)
con = cv2.imread(f"04_contrast/{img_name}", 0)
sha = cv2.imread(f"05_sharpened/{img_name}", 0)

orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

titles = ["Original", "Grayscale", "Denoised", "Contrast", "Sharpened"]
images = [orig_rgb, gray, den, con, sha]

plt.figure(figsize=(12,6))
for i in range(5):
    plt.subplot(2,3,i+1)
    if i == 0:
        plt.imshow(images[i])
    else:
        plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
