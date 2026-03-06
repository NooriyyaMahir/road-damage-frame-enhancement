import cv2
import matplotlib.pyplot as plt
import os

img_name = "frame_200.png"
save_dir = "histograms"
os.makedirs(save_dir, exist_ok=True)

# Load images
orig = cv2.imread(f"01_original_frames/{img_name}")
gray = cv2.imread(f"02_grayscale/{img_name}", 0)
den = cv2.imread(f"03_denoised/{img_name}", 0)
con = cv2.imread(f"04_contrast/{img_name}", 0)
sha = cv2.imread(f"05_sharpened/{img_name}", 0)

orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

images = [orig_rgb, gray, den, con, sha]
hist_images = [orig_gray, gray, den, con, sha]

titles = [
    "Original",
    "Grayscale",
    "Denoised",
    "Contrast",
    "Sharpened"
]

# Create figure with 2 rows and 5 columns
plt.figure(figsize=(20,8))

# ----- FIRST ROW : IMAGES -----
for i in range(5):
    plt.subplot(2,5,i+1)
    if i == 0:
        plt.imshow(images[i])
    else:
        plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

# ----- SECOND ROW : HISTOGRAMS -----
for i in range(5):
    plt.subplot(2,5,i+6)
    plt.hist(hist_images[i].ravel(), bins=256, range=[0,256])
    plt.title(f"{titles[i]} Hist")
    plt.xlabel("Intensity")
    plt.ylabel("Freq")

plt.tight_layout()

# Save final combined figure
plt.savefig(f"{save_dir}/All_Stages_with_Histograms.png", dpi=300)
plt.show()

print("Combined image + histograms saved successfully!")
