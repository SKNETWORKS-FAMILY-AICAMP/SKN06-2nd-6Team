import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def preprocess(self, data_path):
        data = pd.read_csv(data_path, index_col=0)
        # 결측치 삭제
        data.dropna(inplace=True)
        # label encoding
        if "label" in data.columns:
            data["label"] = data["label"].map({"churned": 1, "retained": 0})
        # one hot encoding
        data = pd.get_dummies(data, columns=["device"], dtype=np.float32)
        return data


def set_dataloader(X, y, batch_size=None, mode=None):
    # PyTorch 텐서로 변환 후 TensorDataset과 DataLoader 생성 (배치 학습 적용)
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    if mode == "train":
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        return loader
    if mode == "test":
        loader = DataLoader(dataset, batch_size=2860)
        return loader, y


# 데이터 로드 및 전처리 함수
def load_data(data, learning_type=None, batch_size=None):
    X = data.drop(["label"], axis=1)
    y = data["label"]
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=0
    )
    pd.DataFrame(X_test).to_csv("data/X_test.csv")
    pd.DataFrame(y_test).to_csv("data/y_test.csv")
    # 데이터 표준화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pd.DataFrame(X_test_scaled, index=X_test.index).to_csv("data/X_test_scaled.csv")

    if learning_type == "ml":
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_scaled, y_train, test_size=0.25, stratify=y_train, random_state=0
        )
        return X_train, X_valid, y_train, y_valid, X_test_scaled, y_test

    elif learning_type == "dl":
        y_train = y_train.values
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.values
        y_test = y_test.reshape(-1, 1)
        train_loader = set_dataloader(
            X_train_scaled, y_train, batch_size=batch_size, mode="train"
        )
        test_loader, y_test = set_dataloader(X_test_scaled, y_test, mode="test")
        return train_loader, test_loader, y_test
