import streamlit as st
import pandas as pd
from streamlit_elements import elements, dashboard, mui, nivo
from module.test import metrics


def graphs(metrics_dict, key, mode="metric", graph="Bar", lr="ml"):
    # 그래프 모드
    if mode == "metric":
        metric = metrics_dict["metric"]
        value = metrics_dict["value"]
        nivo_data = []
        for i in range(len(metric)):
            data = {}
            data["metric"] = metric[i]
            value[i] = round(value[i], 4)
            data["pred_test"] = value[i]
            nivo_data.append(data)
    elif mode == "precision-recall":
        precision = metrics_dict["precision"]
        recall = metrics_dict["recall"]
    elif mode == "confusion_matrix":
        pass
    # 그래프

    if graph == "Bar":  # Accuracy, Recall, Precision, F1
        with mui.Box(key=key, sx={"height": 500}, display="flex"):
            # 그래프 표시
            nivo.Bar(
                data=nivo_data,
                keys=["pred_test"],
                indexBy="metric",
                margin={"top": 20, "right": 40, "bottom": 40, "left": 40},
                colors={"scheme": "paired"},
                colorBy="indexValue",
                labelPosition="start",
                theme={
                    "background": "#FFFFFF",
                    "textColor": "#31333F",
                    "tooltip": {
                        "container": {
                            "background": "#FFFFFF",
                            "color": "#33CCFF",
                        }
                    },
                },
            )


def roc(key, lr):
    if lr == "ml":
        image = "image/ML/ROC.png"
    else:
        image = "image/DL/ROC.png"
    with mui.paper(key=key):
        mui.Box(
            component="img",
            src=image,
            alt="ROC Curve",
            sx={"width": "100%", "height": "100%"},
        )


def graph_matrix(y_test, y_pred_list, key):
    with mui.Box(key=key, sx={"height": 500}, display="flex"):
        # MATRIX
        values = y_test.values
        y_actu = pd.Series(values.reshape(-1), name="Actual")
        y_pred = pd.Series(y_pred_list, name="Predicted")
        df_confusion = pd.crosstab(
            y_actu, y_pred, rownames=["Actual"], colnames=["Predicted"]
        )
        df_confusion.columns = ["Churned", "Retained"]
        df_confusion.index = ["Churned", "Retained"]
        heatmap_data = []
        for index, row in df_confusion.iterrows():
            row_data = {
                "id": index,
                "data": [{"x": col, "y": val} for col, val in row.items()],
            }
            heatmap_data.append(row_data)

        nivo.HeatMap(
            data=heatmap_data,
            keys=["x"],
            indexBy="id",
            margin={"top": 50, "right": 20, "bottom": 50, "left": 60},
            axisTop={
                "tickSize": 5,
                "tickPadding": 15,
                "tickRotation": 0,
            },
            axisRight=None,
            axisBottom={
                "tickSize": 5,
                "tickPadding": 15,
                "tickRotation": 0,
            },
            axisLeft={
                "tickSize": 5,
                "tickPadding": 5,
                "tickRotation": 0,
            },
            cellOpacity=1,
            cellBorderColor={"from": "color", "modifiers": [["darker", 0.4]]},
            colors={"type": "quantize", "scheme": "green_blue"},
            hoverTarget="cell",
            labelTextColor="#000000",
        )


def show_dashboard():
    st.header(":bar_chart: Dashboard")
    y_test_path = "data/y_test.csv"

    # ml
    ml_model_path = "model/best_gbm.pkl"
    X_test_path = "data/X_test.csv"
    ml_metrics_dict, y_test, ml_y_pred_list = metrics(
        X_test_path, y_test_path, ml_model_path, mode="ml"
    )

    # dl
    dl_model_path = "model/dl_model_1.pt"
    test_loader_path = "data/test_loader.pth"
    dl_metrics_dict, y_test, dl_y_pred_list = metrics(
        test_loader_path, y_test_path, dl_model_path, mode="dl"
    )
    # 예측기 성능 대시보드
    with elements("dashboard"):
        layout = [
            # Parameters: element_identifier, x_pos, y_pos, width, height
            dashboard.Item("prediction_image_1", 7, 0, 3, 1.5),
            dashboard.Item("prediction_image_2", 7, 4, 3, 1.5),
            dashboard.Item("prediction_image_3", 0, 0, 7, 3),
        ]

        def handle_layout_change(updated_layout):
            return updated_layout

        # ML 평가지표 시각화
        mui.Typography(
            "Machine Learning Model Performance Metrics",
            variant="h6",
        )
        with dashboard.Grid(layout, onLayoutChange=handle_layout_change):
            # 1: Model performance metrics, 2: Loss & Accurcy, 3: ROC Curve, 4: Precision-Recall Curve, 5: Confusion Matrix
            graphs(
                ml_metrics_dict,
                "prediction_image_1",
                graph="Bar",
                mode="metric",
                lr="ml",
            )
            graph_matrix(y_test, ml_y_pred_list, "prediction_image_2")
            image = "https://raw.githubusercontent.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-6Team/refs/heads/main/image/ML/ROC.png"
            mui.Box(
                key="prediction_image_3",
                component="img",
                src=image,
                alt="ROC Curve",
                sx={"width": "100%", "height": "100%"},
            )

        # DL 평가지표 시각화
        mui.Typography(
            "Deep Learning Model Performance Metrics",
            variant="h6",
        )
        with dashboard.Grid(layout, onLayoutChange=handle_layout_change):
            # 1: Model performance metrics, 2: Loss & Accurcy, 3: ROC Curve, 4: Precision-Recall Curve, 5: Confusion Matrix
            graphs(
                dl_metrics_dict,
                "prediction_image_1",
                graph="Bar",
                mode="metric",
                lr="dl",
            )
            graph_matrix(y_test, dl_y_pred_list, "prediction_image_2")
            graphs(
                dl_metrics_dict,
                "prediction_image_3",
                graph="line",
                mode="roc",
                lr="dl",
            )
            image = "https://raw.githubusercontent.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-6Team/refs/heads/main/image/DL/ROC.png"
            mui.Box(
                key="prediction_image_3",
                component="img",
                src=image,
                alt="ROC Curve",
                sx={"width": "100%", "height": "100%"},
            )
