def calculate_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])

    xB = min(
        boxA[0] + boxA[2],
        boxB[0] + boxB[2]
    )

    yB = min(
        boxA[1] + boxA[3],
        boxB[1] + boxB[3]
    )

    interArea = (
        max(0, xB - xA) *
        max(0, yB - yA)
    )

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    unionArea = (
        boxAArea +
        boxBArea -
        interArea
    )

    return interArea / (unionArea + 1e-5)


def calculate_metrics(TP, FP, FN, TN):

    accuracy = (
        (TP + TN) /
        (TP + TN + FP + FN + 1e-5)
    )

    precision = (
        TP /
        (TP + FP + 1e-5)
    )

    recall = (
        TP /
        (TP + FN + 1e-5)
    )

    f1_score = (
        2 * precision * recall /
        (precision + recall + 1e-5)
    )

    return accuracy, precision, recall, f1_score
