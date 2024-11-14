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
            # Parameters: element_identifier, x_pos, y_pos, width, height, [item properties...]
            dashboard.Item("first_item", 0, 0, 5, 2),
            dashboard.Item("second_item", 5, 0, 5, 2),
            dashboard.Item("third_item", 0, 2, 5, 1),
            dashboard.Item("fourth_item", 5, 2, 5, 1),
            dashboard.Item("plot", 0, 10, 5, 5),
        ]

        def handle_layout_change(updated_layout):
            return updated_layout

        with dashboard.Grid(layout, onLayoutChange=handle_layout_change):
            mui.Paper("First item", key="first_item")
            mui.Paper("Second item", key="second_item")
            mui.Paper("Third item", key="third_item")
            mui.Paper("Fourth item", key="fourth_item")

            with mui.Box(key="plot", sx={"height": 500}, display="flex"):
                nivo.Radar(
                    data=data,
                    keys=columns,
                    indexBy="label",
                    valueFormat=">-.2f",
                    margin={"top": 70, "right": 80, "bottom": 40, "left": 80},
                    borderColor={"from": "color"},
                    gridLabelOffset=36,
                    dotSize=10,
                    dotColor={"theme": "background"},
                    dotBorderWidth=2,
                    motionConfig="wobbly",
                    legends=[
                        {
                            "anchor": "top-left",
                            "direction": "column",
                            "translateX": -50,
                            "translateY": -40,
                            "itemWidth": 80,
                            "itemHeight": 20,
                            "itemTextColor": "#999",
                            "symbolSize": 12,
                            "symbolShape": "circle",
                            "effects": [
                                {"on": "hover", "style": {"itemTextColor": "#000"}}
                            ],
                        }
                    ],
                    theme={
                        "background": "#FFFFFF",
                        "textColor": "#31333F",
                        "tooltip": {
                            "container": {
                                "background": "#FFFFFF",
                                "color": "#31333F",
                            }
                        },
                    },
                )
