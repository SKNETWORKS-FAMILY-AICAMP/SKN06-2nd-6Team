import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class Preprocessor:
    def __init__(self, data_path):
        self.data_path = data_path

    def preprocess(self):
        data = pd.read_csv(self.data_path, index_col=0)
        # 결측치 삭제
        data.dropna(inplace=True)
        # label encoding
        l_encoder = LabelEncoder()
        if "label" in data.columns:
            data["label"] = l_encoder.fit_transform(data["label"])
        # one hot encoding
        data = pd.get_dummies(data, columns=["device"])
        return data


def set_dataloader(X, y, batch_size=None, mode=None):
    # PyTorch 텐서로 변환 후 TensorDataset과 DataLoader 생성 (배치 학습 적용)
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    if mode == "train":
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader
    if mode == "test":
        loader = DataLoader(dataset, batch_size=2860)
        return loader, y


# 데이터 로드 및 전처리 함수
def load_data(data, learning_type=None, batch_size=None):
    X = data.drop(["label"], axis=1)
    if learning_type == "ml":
        y = data["label"]
    elif learning_type == "dl":
        y = np.array(data["label"])
        y = y.reshape(-1, 1)
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=0
    )

    # 데이터 표준화
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if learning_type == "ml":
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.25, stratify=y_train, random_state=0
        )
        return X_train, X_valid, y_train, y_valid, X_test, y_test

    elif learning_type == "dl":
        train_loader = set_dataloader(
            X_train, y_train, batch_size=batch_size, mode="train"
        )
        test_loader, y_test = set_dataloader(X_test, y_test, mode="test")
        return train_loader, test_loader, y_test
