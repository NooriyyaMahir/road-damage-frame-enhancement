import cv2
import numpy as np

# Load predicted mask
pred = cv2.imread("output/morphology/frame_0.jpg", 0)

# Load ground truth mask
gt = cv2.imread("ground_truth/gt_0.jpg", 0)

# Resize if needed
gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))

# Binary conversion
_, pred = cv2.threshold(pred, 127, 1, cv2.THRESH_BINARY)
_, gt = cv2.threshold(gt, 127, 1, cv2.THRESH_BINARY)

# Calculate TP, TN, FP, FN
tp = np.sum((pred == 1) & (gt == 1))
tn = np.sum((pred == 0) & (gt == 0))
fp = np.sum((pred == 1) & (gt == 0))
fn = np.sum((pred == 0) & (gt == 1))

# Metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)

precision = tp / (tp + fp + 1e-6)

recall = tp / (tp + fn + 1e-6)

f1 = (2 * precision * recall) / (precision + recall + 1e-6)

iou = tp / (tp + fp + fn + 1e-6)

dice = (2 * tp) / ((2 * tp) + fp + fn + 1e-6)

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-score :", round(f1, 4))
print("IoU      :", round(iou, 4))
print("Dice     :", round(dice, 4))
