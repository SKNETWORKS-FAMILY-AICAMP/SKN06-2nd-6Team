import torch
import torch.nn as nn


# 모델
class ChurnPredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Linear(12, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.32)
        )
        self.b2 = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.32)
        )
        self.b3 = nn.Sequential(
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.32)
        )
        self.b4 = nn.Sequential(
            nn.Linear(32, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Dropout(0.32)
        )
        self.b5 = nn.Linear(8, 1)
        self.logistic = nn.Sigmoid()

    def forward(self, x):
        # 각 은닉층 통과 시 배치 정규화 및 드롭아웃 적용
        x = nn.Flatten()(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.logistic(x)  # 이진 분류를 위한 sigmoid 활성화 함수
        return x
