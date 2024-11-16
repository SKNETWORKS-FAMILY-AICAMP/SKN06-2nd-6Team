import sys
import streamlit as st
import pandas as pd
import joblib
import torch
from torch import nn
import numpy as np


# sys.path.append('/Users/haeun/Desktop/SKN2nd_bhe/module')
# from dl_model import ChurnPredictionModel

# Load models
best_gbm = joblib.load('model/best_gbm.pkl')

try:
    dl_model = torch.load('model/dl_model_outlier.pth', map_location=torch.device('cpu'))
    dl_model.eval()  # Set model to evaluation mode
except AttributeError:
    st.error("Deep Learning Model could not be loaded. Ensure the ChurnPredictionModel class is properly defined.")

# Define function for deep learning model prediction
def dl_predict(model, inputs):
    with torch.no_grad():
        inputs = torch.tensor(inputs, dtype=torch.float32)
        return model(inputs).numpy()

# Streamlit UI
def show_predictor():
    st.header(":bookmark_tabs: Customer Churn Prediction Service")

    # dataset
    data = pd.read_csv("data/sample_data.csv", index_col=0)
    st.write("Loaded Dataset:")
    st.write(data)

    if 'device' in data.columns:
        # Assume 'device' values need to be mapped to 1 or 0
        data['device'] = data['device'].apply(lambda x: 1 if x == 'iPhone' else 0)  # Adjust condition as needed
    else:
        st.error("'device' column not found in the dataset.")
        return
    
    st.write("Preprocessed Dataset:")
    st.write(data)

    # Choose model for prediction
    model_choice = st.selectbox("Choose the prediction model:", ("Gradient Boosting Machine (GBM)", "Deep Learning Model"))

    if st.button("Predict"):
        # Prepare data for prediction
        features = data.values


        if model_choice == "Gradient Boosting Machine (GBM)":
            predictions = best_gbm.predict(features)
            st.write("Predictions (GBM):")
            st.write(predictions)

        elif model_choice == "Deep Learning Model":
            predictions = dl_predict(dl_model, features)
            st.write("Predictions (Deep Learning Model):")
            st.write(predictions)

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
