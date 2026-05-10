import cv2
import os

# Video
video_path = "dataset/V1.mp4"

cap = cv2.VideoCapture(video_path)

# Output folders
os.makedirs("output/enhanced", exist_ok=True)
os.makedirs("output/segmented", exist_ok=True)
os.makedirs("output/final", exist_ok=True)

frame_id = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # Enhancement
    enhanced = enhance_frame(frame)

    # Segmentation
    edges, segmented = segment_damage(enhanced)

    # Detection
    final_output = detect_damage(frame, segmented)

    # Save frames
    if frame_id % 10 == 0:

        cv2.imwrite(f"output/enhanced/frame_{frame_id}.jpg", enhanced)
        cv2.imwrite(f"output/segmented/frame_{frame_id}.jpg", segmented)
        cv2.imwrite(f"output/final/frame_{frame_id}.jpg", final_outp)

    # Display
    cv2.imshow("Final Output", final_output)

    frame_id += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
