import streamlit as st
import pandas as pd
from streamlit_elements import elements, dashboard, mui, nivo


def member_image(handle_layout_change):
    # 멤버별 이미지
    layout = [
        # Parameters: element_identifier, x_pos, y_pos, width, height
        dashboard.Item("team", 0, 0, 8, 0.2, isResizable=False),
        dashboard.Item("period", 8, 0, 2, 0.2, isResizable=False),
        dashboard.Item("member_image_1", 0, 2, 2.5, 1.5, isResizable=False),
        dashboard.Item("member_image_2", 2.5, 2, 2.5, 1.5, isResizable=False),
        dashboard.Item("member_image_3", 5, 2, 2.5, 1.5, isResizable=False),
        dashboard.Item("member_image_4", 7.5, 2, 2.5, 1.5, isResizable=False),
    ]
    with dashboard.Grid(layout, onLayoutChange=handle_layout_change):
        with mui.Box(key="team"):
            mui.Typography(
                "6Team 아무래도 귀엽조", variant="h6", sx={"color": "#000000"}
            )
        with mui.Box(key="period"):
            mui.Typography(
                "2024.11.13 ~ 2024.11.14", variant="h7", sx={"color": "#000000"}
            )

        # 1: 승현님, 2: 서윤님, 3: 하은님 4: 코딩 에러 으아아
        image_1 = "https://raw.githubusercontent.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-6Team/refs/heads/main/image/%EC%8A%B9%ED%98%84%EB%8B%98%EB%84%A4%EA%B7%80%EC%9A%94%EB%AF%B8.png"
        image_2 = "https://raw.githubusercontent.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-6Team/refs/heads/main/image/%EC%84%9C%EC%9C%A4%EB%8B%98%EB%84%A4%EA%B7%80%EC%9A%94%EB%AF%B8.png"
        image_3 = "https://raw.githubusercontent.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-6Team/refs/heads/main/image/%ED%95%98%EC%9D%80%EB%8B%98%EB%84%A4%EA%B7%80%EC%9A%94%EB%AF%B8.png"
        image_4 = "https://raw.githubusercontent.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-6Team/refs/heads/main/image/%EB%B6%80%EC%8B%A4%EA%B0%90%EC%9E%90.jpeg"
        with mui.Paper(key="member_image_2"):
            mui.Box(
                component="img",
                src=image_1,
                alt="김승현",
                sx={"width": "100%", "height": "100%"},
            ),
            mui.Box("김승현/DL")
        with mui.Paper(key="member_image_4"):
            mui.Box(
                component="img",
                src=image_2,
                alt="박서윤",
                sx={"width": "100%", "height": "100%"},
            ),
            mui.Box("박서윤/ML")
        with mui.Paper(key="member_image_3"):
            mui.Box(
                component="img",
                src=image_3,
                alt="백하은",
                sx={"width": "100%", "height": "100%"},
            ),
            mui.Box("백하은/결과보고서, ML, streamlit")
        with mui.Paper(key="member_image_1"):
            mui.Box(
                component="img",
                src=image_4,
                alt="정유진",
                sx={"width": "100%", "height": "100%"},
            ),
            mui.Box("정유진/전처리, DL, streamlit")


def show_team():
    # 멤버별 이미지
    with elements("dashboard"):

        def handle_layout_change(updated_layout):
            return updated_layout

        member_image(handle_layout_change)
