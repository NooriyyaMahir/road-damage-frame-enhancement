import cv2
import numpy as np
import os

# VIDEO PATH
video_path = 'V1.mp4'
cap = cv2.VideoCapture(video_path)

# OUTPUT FOLDERS
os.makedirs("output/original", exist_ok=True)
os.makedirs("output/blur", exist_ok=True)
os.makedirs("output/edges", exist_ok=True)
os.makedirs("output/final", exist_ok=True)


# DAMAGE CLASSIFICATION
def get_color(cnt, frame_width):

    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)

    aspect_ratio = w / float(h + 1e-5)
    extent = area / (w * h + 1e-5)

    perimeter = cv2.arcLength(cnt, True)
    circularity = (4 * np.pi * area) / (perimeter * perimeter + 1e-5)


    yellow = (0, 255, 255)

    # Pothole
    if area > 5000 and circularity > 0.45:
        return yellow, "Pothole"
    
    # Alligator
    elif area > 3000 and extent > 0.5 and circularity < 0.45:
        return yellow, "Alligator"
    
    # Edge Crack
    elif x < 50 or (x + w) > (frame_width - 50):
        return yellow, "Edge Crack"
    
    # Raveling
    elif area < 3000 and extent < 0.5:
        return yellow, "Raveling"
    
    # Crack (thin + small only)
    elif area < 3000 and (aspect_ratio > 3 or aspect_ratio < 0.3):
        return yellow, "Crack"
    
    # Surface
    else:
        return yellow, "Surface"


# CHECK VIDEO
if not cap.isOpened():
    print("Error opening video file")

else:

    width, height = 640, 480

    # OUTPUT VIDEO
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter("damage_output.avi", fourcc, 30, (width, height))

    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # RESIZE FRAME
        frame = cv2.resize(frame, (width, height))

        output = frame.copy()

    
        # PREPROCESSING
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Gaussian Blur
        blur = cv2.GaussianBlur(gray, (5, 5), 1)

        # EDGE DETECTION
        edges = cv2.Canny(blur, 50, 150)

        # MORPHOLOGY
        kernel = np.ones((5, 5), np.uint8)

        closed = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        closed = cv2.dilate(
            closed,
            np.ones((3, 3), np.uint8),
            iterations=1
        )

        
        # FIND CONTOURS
        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

    
        # DRAW DETECTION
        for cnt in contours:

            area = cv2.contourArea(cnt)

            # REMOVE SMALL NOISE
            if area < 1000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            if w > 20 and h > 20:

                color, label = get_color(cnt, width)

                # DRAW CONTOUR
                cv2.drawContours(output, [cnt], -1, color, 2)


        
        # SAVE FRAMES
        if frame_id % 10 == 0:

            cv2.imwrite(f"output/original/frame_{frame_id}.jpg", frame)
            cv2.imwrite(f"output/blur/frame_{frame_id}.jpg", blur)
            cv2.imwrite(f"output/edges/frame_{frame_id}.jpg", edges)
            cv2.imwrite(f"output/final/frame_{frame_id}.jpg", output)

        frame_id += 1

        # DISPLAY OUTPUT
        cv2.imshow("Original video", frame)
        cv2.imshow("Road Damage Detection", output)
        

        out.write(output)

        # PRESS Q TO EXIT
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

   
    # RELEASE
    cap.release()
    out.release()

    cv2.destroyAllWindows()
  
