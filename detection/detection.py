import cv2
import numpy as np

def classify_damage(cnt, frame_width):

    area = cv2.contourArea(cnt)

    x, y, w, h = cv2.boundingRect(cnt)

    aspect_ratio = w / float(h + 1e-5)

    extent = area / (w * h + 1e-5)

    perimeter = cv2.arcLength(cnt, True)

    circularity = (
        4 * np.pi * area
    ) / (perimeter * perimeter + 1e-5)

    if area > 5000 and circularity > 0.45:
        return "Pothole"

    elif area > 3000 and extent > 0.5:
        return "Alligator"

    elif x < 50 or (x + w) > (frame_width - 50):
        return "Edge Crack"

    elif area < 3000 and extent < 0.5:
        return "Raveling"

    elif area < 3000 and (
        aspect_ratio > 3 or aspect_ratio < 0.3
    ):
        return "Crack"

    return "Surface"


def detect_damage(frame, segmented):

    output = frame.copy()

    contours, _ = cv2.findContours(
        segmented,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 1000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if w > 20 and h > 20:

            label = classify_damage(cnt, frame.shape[1])

            color = (0, 255, 255)

            cv2.drawContours(
                output,
                [cnt],
                -1,
                color,
                2
            )

            cv2.putText(
                output,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    return output
