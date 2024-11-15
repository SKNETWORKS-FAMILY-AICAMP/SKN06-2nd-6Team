import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_elements import elements, dashboard, mui, editor, media, lazy, sync, nivo


def show_dashboard():
    st.header(":bar_chart: Dashboard")
    data = pd.read_csv("data/waze_dataset.csv")
    columns = data.columns.drop("label")

    with elements("dashboard"):
        layout = [
            # Parameters: element_identifier, x_pos, y_pos, width, height
            dashboard.Item("first_item", 0, 0, 2, 2),
            dashboard.Item("second_item", 2, 0, 2, 2),
            dashboard.Item("third_item", 4, 0, 2, 2),
            dashboard.Item("fourth_item", 6, 0, 2, 2),
        ]

        def handle_layout_change(updated_layout):
            return updated_layout

        # 1: 하은님 강아지, 2: 승현님 강아지, 3: 서윤님 인형, 4: 코딩 에러 으아아
        image_1 = "https://raw.githubusercontent.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-6Team/refs/heads/main/image/%ED%95%98%EC%9D%80%EB%8B%98%EB%84%A4%EA%B7%80%EC%9A%94%EB%AF%B8.png"
        image_2 = "https://raw.githubusercontent.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-6Team/refs/heads/main/image/%EC%8A%B9%ED%98%84%EB%8B%98%EB%84%A4%EA%B7%80%EC%9A%94%EB%AF%B8.png"
        image_3 = "https://raw.githubusercontent.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-6Team/refs/heads/main/image/%EC%84%9C%EC%9C%A4%EB%8B%98_%EC%95%A0%EC%B0%A9%EC%9D%B8%ED%98%95.png"
        image_4 = "https://raw.githubusercontent.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-6Team/refs/heads/main/image/%EB%B6%80%EC%8B%A4%EA%B0%90%EC%9E%90.jpeg"
        with dashboard.Grid(layout, onLayoutChange=handle_layout_change):
            with mui.Paper(elevation=3, key="first_item"):
                mui.Box(
                    component="img",
                    src=image_1,
                    alt="Correlation Image",
                    sx={"width": "100%", "height": "100%"},
                )
            with mui.Paper(elevation=3, key="second_item"):
                mui.Box(
                    component="img",
                    src=image_2,
                    alt="Correlation Image",
                    sx={"width": "100%", "height": "100%"},
                )
            with mui.Paper(elevation=3, key="third_item"):
                mui.Box(
                    component="img",
                    src=image_3,
                    alt="Correlation Image",
                    sx={"width": "100%", "height": "100%"},
                )
            with mui.Paper(elevation=3, key="fourth_item"):
                mui.Box(
                    component="img",
                    src=image_4,
                    alt="Correlation Image",
                    sx={"width": "100%", "height": "100%"},
                )
