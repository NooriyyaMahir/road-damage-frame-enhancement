import cv2
import numpy as np

def segment_damage(blur):

    # Edge detection
    edges = cv2.Canny(enhanced, 50, 150)

    # Morphology kernel
    kernel = np.ones((5, 5), np.uint8)

    # Close gaps
    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Dilate segmented regions
    segmented = cv2.dilate(
        closed,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    return edges, segmented
