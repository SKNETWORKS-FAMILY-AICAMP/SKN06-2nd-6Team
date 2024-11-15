import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_elements import elements, dashboard, mui, editor, media, lazy, sync, nivo
from sklearn.metrics import confusion_matrix

from module.dl_test import metrics


def member_image(layout, handle_layout_change):
    # 멤버별 이미지
    # 1: 승현님, 2: 서윤님, 3: 하은님 4: 코딩 에러 으아아
    image_1 = "https://raw.githubusercontent.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-6Team/refs/heads/main/image/%EC%8A%B9%ED%98%84%EB%8B%98%EB%84%A4%EA%B7%80%EC%9A%94%EB%AF%B8.png"
    image_2 = "https://raw.githubusercontent.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-6Team/refs/heads/main/image/%EC%84%9C%EC%9C%A4%EB%8B%98%EB%84%A4%EA%B7%80%EC%9A%94%EB%AF%B8.png"
    image_3 = "https://raw.githubusercontent.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-6Team/refs/heads/main/image/%ED%95%98%EC%9D%80%EB%8B%98%EB%84%A4%EA%B7%80%EC%9A%94%EB%AF%B8.png"
    image_4 = "https://raw.githubusercontent.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-6Team/refs/heads/main/image/%EB%B6%80%EC%8B%A4%EA%B0%90%EC%9E%90.jpeg"
    with dashboard.Grid(layout, onLayoutChange=handle_layout_change):
        with mui.Paper(key="member_image_1"):
            mui.Box(
                component="img",
                src=image_1,
                alt="김승현",
                sx={"width": "100%", "height": "100%"},
            ),
            mui.Box("이름: 김승현", sx={"width": "100%"})
        with mui.Paper(key="member_image_2"):
            mui.Box(
                component="img",
                src=image_2,
                alt="박서윤",
                sx={"width": "100%", "height": "100%"},
            ),
            mui.Box("이름: 박서윤", sx={"width": "100%"})
        with mui.Paper(key="member_image_3"):
            mui.Box(
                component="img",
                src=image_3,
                alt="백하은",
                sx={"width": "100%", "height": "100%"},
            ),
            mui.Box("이름: 백하은", sx={"width": "100%"})
        with mui.Paper(key="member_image_4"):
            mui.Box(
                component="img",
                src=image_4,
                alt="정유진",
                sx={"width": "100%", "height": "100%"},
            ),
            mui.Box("이름: 정유진", sx={"width": "100%"})


def show_dashboard():
    st.header(":bar_chart: Dashboard")
    model_path = "model/dl_model.pt"
    test_loader_path = "data/test_loader.pth"
    y_test_path = "data/y_test.csv"
    metrics_dict, _, _ = metrics(
        test_loader_path, y_test_path, model_path, device="cpu"
    )

    # 예측기 성능 대시보드
    with elements(key="prediction"):
        layout = [
            # Parameters: element_identifier, x_pos, y_pos, width, height
            dashboard.Item("prediction_image_1", 0, 0, 5, 3),
            dashboard.Item("prediction_image_2", 5, 0, 5, 3),
            dashboard.Item("prediction_image_3", 0, 4, 5, 3),
            dashboard.Item("prediction_image_4", 5, 4, 5, 3),
            dashboard.Item("member_image_1", 0, 5, 2.5, 1.5, isResizable=False),
            dashboard.Item("member_image_2", 2.5, 5, 2.5, 1.5, isResizable=False),
            dashboard.Item("member_image_3", 5, 5, 2.5, 1.5, isResizable=False),
            dashboard.Item("member_image_4", 7.5, 5, 2.5, 1.5, isResizable=False),
        ]

        def handle_layout_change(updated_layout):
            return updated_layout

        metric = metrics_dict["metric"]
        value = metrics_dict["value"]
        performance_metrics = pd.DataFrame(metric, index=value)

        # 1: Model performance metrics, 2: Loss & Accurcy, 3: ROC Curve, 4: Precision-Recall Curve, 5: Confusion Matrix
        with mui.Paper(key="prediction_image_1"):
            nivo.Bar(
                data=performance_metrics,
                keys=["value"],
                indexBy="metric",
                margin={"top": 50, "right": 130, "bottom": 50, "left": 60},
                padding=0.3,
                valueScale={"type": "linear"},
                indexScale={"type": "band", "round": True},
                colors="nivo",
            )
        # 멤버별 이미지
        member_image(layout, handle_layout_change)
