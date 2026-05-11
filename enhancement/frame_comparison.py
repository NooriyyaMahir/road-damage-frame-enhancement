import cv2
import matplotlib.pyplot as plt

img_name = "frame_200.png"  

orig = cv2.imread(f"01_original_frames/{img_name}")
gray = cv2.imread(f"02_grayscale/{img_name}", 0)
den = cv2.imread(f"03_denoised/{img_name}", 0)

orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

titles = ["Original", "Grayscale", "Denoised"]
images = [orig_rgb, gray, den]

plt.figure(figsize=(12,6))
for i in range(3):
    plt.subplot(2,3,i+1)
    if i == 0:
        plt.imshow(images[i])
    else:
        plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
