import numpy as np
import torch

def test(test_loader, model, save_path, device="cpu"):
    # load best model
    best_model = torch.load(save_path, weights_only=False)
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
            valid_acc += torch.sum((pred_test < 0.5).type(torch.int32) == y_batch).item()
            y_pred_list.append(pred_test)
            y_prob_list.append(prob_test)
        y_pred_list = np.concatenate(y_pred_list)
        y_prob_list = np.concatenate(y_prob_list)
    return y_pred_list, y_prob_list
