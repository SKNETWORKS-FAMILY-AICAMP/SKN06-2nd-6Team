import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time

# train 함수에 들어가는 valid 부분
def valid(dataloader, model, loss_fn, device="cpu"):
    model.to(device)
    model.eval()
    loss = acc = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)  # positive일 확률
            loss += loss_fn(pred, y).item()
            #  이진 분류에서 accuracy
            acc += torch.sum((pred > 0.5).type(torch.int32) == y).item()
        loss /= len(dataloader)
        acc /= len(dataloader.dataset)
    return loss, acc

def train(dataloader, model, loss_fn, optimizer, scheduler, device="cpu"):
    model.train()
    train_loss = 0
    # 학습률 조정 스케줄러: CosineAnnealingWarmRestarts를 사용하여 학습률을 주기적으로 리셋
    scheduler = scheduler

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        # 손실 계산 및 역전파
        loss = loss_fn(pred, y)
        loss.backward()
        # 파라미터 업데이트
        optimizer.step()
        # 파라미터 초기화
        optimizer.zero_grad()
        train_loss += loss.item()
    # LR 변경 요청
    scheduler.step()
    # loss 계산 및 list 추가
    train_loss /= len(dataloader)
    return train_loss

# 학습
def fit(num_epochs, train_loader, valid_loader, model, loss_fn, optimizer, scheduler, save_path, patience, device="cpu"):
    # 손실과 정확도 기록용 리스트 초기화
    train_losses = []
    valid_losses = []
    valid_acces = []
    best_score = torch.inf  # 성능 개선 기준 초기값
    trigger_count = 0  # 개선이 없을 때 증가하는 카운트

    s = time.time()  # 학습 시작 시간 기록
    for epoch in range(num_epochs):
        # 모델 학습
        train_loss = train(train_loader, model, loss_fn, optimizer, scheduler)
        # 모델 검증
        valid_loss, valid_acc = valid(valid_loader, model, loss_fn)
        
        # loss 계산 및 list 추가
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_acces.append(valid_acc)

        # Best Score 업데이트 및 조기 종료 체크
        if valid_loss < best_score:
            print(
                f"Epoch [{epoch+1:3d}/{num_epochs}], >>>>> Loss Improved from {best_score:.4f} to {valid_loss:.4f}. Saving Model."
            )
            best_score = valid_loss
            torch.save(model, save_path)
            trigger_count = 0  # 성능 개선 시 초기화
        else:
            trigger_count += 1
            if trigger_count >= patience:
                print(
                    f"Early stopping at epoch {epoch+1:3d}. No improvement for {patience} epochs."
                )
                break

        print(
            f"Epoch [{epoch+1:3d}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}"
        )

    e = time.time()
    training_time = e - s
    print(f"\nTotal Training Time: {training_time:.2f} seconds")
    return train_losses, valid_losses, valid_acces
