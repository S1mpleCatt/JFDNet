import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def multilabel_metrics(labels, probabilities, evaluated_joints=4, threshold=0.5):
    labels = np.asarray(labels)[:, :evaluated_joints].astype(np.int64)
    probabilities = np.asarray(probabilities)[:, :evaluated_joints]
    predictions = (probabilities >= threshold).astype(np.int64)
    per_joint = []
    for index in range(evaluated_joints):
        target = labels[:, index]
        predicted = predictions[:, index]
        per_joint.append(
            {
                "joint": index + 1,
                "accuracy": float(accuracy_score(target, predicted)),
                "precision": float(precision_score(target, predicted, zero_division=0)),
                "recall": float(recall_score(target, predicted, zero_division=0)),
                "f1": float(f1_score(target, predicted, zero_division=0)),
            }
        )
    return {
        "accuracy": float(np.mean([item["accuracy"] for item in per_joint])),
        "precision": float(np.mean([item["precision"] for item in per_joint])),
        "recall": float(np.mean([item["recall"] for item in per_joint])),
        "f1": float(np.mean([item["f1"] for item in per_joint])),
        "per_joint": per_joint,
    }
