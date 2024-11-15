import torch
import torch.nn as nn
# 모델
class ChurnPredictionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 은닉층과 배치 정규화, 드롭아웃 추가
        self.fc1 = nn.Linear(12, 128)  # 첫 번째 은닉층
        self.bn1 = nn.BatchNorm1d(128)  # 첫 번째 배치 정규화
        self.fc2 = nn.Linear(128, 64)  # 두 번째 은닉층
        self.bn2 = nn.BatchNorm1d(64)  # 두 번째 배치 정규화
        self.fc3 = nn.Linear(64, 32)  # 세 번째 은닉층
        self.bn3 = nn.BatchNorm1d(32)  # 세 번째 배치 정규화
        self.fc4 = nn.Linear(32, 1)  # 출력층

        # 활성화 함수와 드롭아웃 설정
        self.relu = nn.ReLU()  # 활성화 함수
        self.dropout = nn.Dropout(0.35)  # 드롭아웃 확률: 35%

    def forward(self, x):
        # 각 은닉층 통과 시 배치 정규화 및 드롭아웃 적용
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        x = torch.sigmoid(self.fc4(x))  # 이진 분류를 위한 sigmoid 활성화 함수
        return x
