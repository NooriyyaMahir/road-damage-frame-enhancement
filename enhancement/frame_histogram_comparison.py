import cv2
import matplotlib.pyplot as plt
import os

img_name = "frame_200.png"
save_dir = "histograms"
os.makedirs(save_dir, exist_ok=True)


# LOAD IMAGES
orig = cv2.imread(f"01_original_frames/{img_name}")
gray = cv2.imread(f"02_grayscale/{img_name}", 0)
den = cv2.imread(f"03_denoised/{img_name}", 0)

# Convert original for display
orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)


# IMAGE LISTS
images = [orig_rgb, gray, den]

hist_images = [orig_gray, gray, den]

titles = [
    "Original",
    "Grayscale",
    "Denoised"
]

# Number of stages
n = len(images)

# CREATE FIGURE
plt.figure(figsize=(15, 8))

# FIRST ROW : IMAGES
for i in range(n):

    plt.subplot(2, n, i + 1)

    if i == 0:
        plt.imshow(images[i])
    else:
        plt.imshow(images[i], cmap='gray')

    plt.title(titles[i])
    plt.axis('off')


# SECOND ROW : HISTOGRAMS
for i in range(n):

    plt.subplot(2, n, i + n + 1)

    plt.hist(
        hist_images[i].ravel(),
        bins=256,
        range=[0,256]
    )

    plt.title(f"{titles[i]} Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")


# SAVE OUTPUT
plt.tight_layout()

save_path = f"{save_dir}/All_Stages_with_Histograms.png"

plt.savefig(save_path, dpi=300)

plt.show()

print("Combined image + histograms saved successfully!")
print("Saved at:", save_path)
