import cv2
import os

video_path = "video_1.mp4"
out_dir = "original_frames"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if count % 10 == 0:  # save every 10th frame
        cv2.imwrite(f"{out_dir}/frame_{count}.png", frame)

    count += 1

cap.release()
print("Original frames extracted")
