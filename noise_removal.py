import cv2

def remove_noise(image):

    median = cv2.medianBlur(image, 5)

    gaussian = cv2.GaussianBlur(median, (5,5), 0)

    return gaussian