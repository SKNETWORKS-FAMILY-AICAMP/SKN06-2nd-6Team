import sys
import streamlit as st
import pandas as pd
import joblib
import torch
from torch import nn
import numpy as np
from module.dl_model import ChurnPredictionModel
from module.dataload import Preprocessor


# Load models
best_gbm = joblib.load("model/best_gbm.pkl")

try:
    dl_model = torch.load("model/dl_model.pt", map_location=torch.device("cpu"))
    dl_model.eval()  # Set model to evaluation mode
except AttributeError:
    st.error(
        "Deep Learning Model could not be loaded. Ensure the ChurnPredictionModel class is properly defined."
    )


# Define function for deep learning model prediction
def dl_predict(model, inputs):
    inputs = torch.tensor(inputs, dtype=torch.float32)
    device = torch.device("cpu")
    with torch.no_grad():
        inputs = inputs.to(device)
        y_pred = model(inputs)
        y_pred = (y_pred < 0.5).type(torch.int32)
        return y_pred


# Streamlit UI
def show_predictor():
    st.header(":bookmark_tabs: Customer Churn Prediction Service")

    # dataset
    data = pd.read_csv("data/sample_data.csv", index_col=0)
    st.subheader("Loaded Dataset:", divider=True)
    st.write(data)

    data = Preprocessor().preprocess("data/sample_data.csv")

    st.subheader("Preprocessed Dataset:", divider=True)
    st.write(data)

    # Choose model for prediction
    st.subheader("Prediction", divider=True)
    model_choice = st.selectbox(
        "Choose the prediction model:",
        ("Gradient Boosting Machine (GBM)", "Deep Learning Model"),
    )

    if st.button("Predict"):
        # Prepare data for prediction
        features = data.values

        if model_choice == "Gradient Boosting Machine (GBM)":
            predictions = best_gbm.predict(features)
        elif model_choice == "Deep Learning Model":
            predictions = dl_predict(dl_model, features)

        predictions = pd.DataFrame(predictions, columns=["Churn"], index=data.index)
        predictions["Churn"] = predictions["Churn"].map({1: "No", 0: "Yes"})
        yes = predictions[predictions["Churn"] == "Yes"]
        no = predictions[predictions["Churn"] == "No"]
        col1, col2 = st.columns(2)
        with col1:
            st.write("Yes:", yes.shape[0])
            st.dataframe(yes, use_container_width=True)
        with col2:
            st.write("No:", no.shape[0])
            st.dataframe(no, use_container_width=True)


if __name__ == "__main__":
    show_predictor()


# 데이터 로드

# 전처리
# 임포트
# rand


# 모델 로드
# ml 모델
# 돌리고
# 결과값 받고


# dl 모델
# 돌리고
# 결과값받고

# 결과 출력
# 이 사람이 이탈할 것이다/아니다
