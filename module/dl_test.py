import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    roc_curve,
    auc,
    precision_recall_curve,
)


def test(test_loader, y_test, best_model, device="cpu"):
    # load best model
    best_model.to(device)
    best_model.eval()
    # 모델 평가
    y_pred_list = []
    y_prob_list = []
    valid_acc = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            prob_test = best_model(X_batch)
            pred_test = prob_test.round()
            valid_acc += torch.sum(
                (pred_test < 0.5).type(torch.int32) == y_batch
            ).item()
            y_pred_list.append(pred_test)
            y_prob_list.append(prob_test)
        y_pred_list = np.concatenate(y_pred_list)
        y_prob_list = np.concatenate(y_prob_list)
    return y_pred_list, y_prob_list

def metrics(test_loader_path, y_test_path, model_path, device="cpu"):
    test_loader = torch.load(test_loader_path)
    y_test = np.loadtxt(y_test_path)
    best_model = torch.load(model_path, map_location="cpu")
    y_pred_list, y_prob_list = test(test_loader, y_test, best_model, device="cpu")
    metrics = {}
    # 성능 평가 지표 계산
    accuracy = accuracy_score(y_test, y_pred_list)
    recall = recall_score(y_test, y_pred_list)
    precision = precision_score(y_test, y_pred_list)
    f1 = f1_score(y_test, y_pred_list)
    metrics["metric"] = ["Accuracy", "Recall", "Precision", "F1 Score"]
    metrics["value"] = [accuracy, recall, precision, f1]
    # roc curve
    fpr, tpr, _ = roc_curve(y_test, y_prob_list)
    roc_auc = auc(fpr, tpr)
    metrics["fpr"] = fpr
    metrics["tpr"] = tpr
    metrics["roc_auc"] = roc_auc

    # precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob_list)
    metrics["precision"] = precision
    metrics["recall"] = recall

    return metrics, y_test, y_pred_list
