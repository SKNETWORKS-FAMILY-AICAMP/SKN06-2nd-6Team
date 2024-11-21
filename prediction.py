import streamlit as st
from streamlit_option_menu import option_menu
from views import data, about, predictor, dashboard, team

# page config
st.set_page_config(
    page_title="Waze 이탈 예측",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 사이드 바 생성
with st.sidebar:
    choose = option_menu(
        menu_title="Churn prediction",
        options=["Team", "About", "Data", "Predictor", "Dash board"],
        icons=[
            "bi bi-people",
            "bi bi-cursor",
            "bi bi-archive",
            "bi bi-graph-up-arrow",
            "bi bi-bar-chart",
        ],
        menu_icon="house",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#fafafa"},
            "icon": {"color": "black", "font-size": "18px"},
            "nav-link": {
                "font-size": "13px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#02ab21"},
        },
    )
######################################################

# About Page
if choose == "Team":
    team.show_team()
elif choose == "About":
    about.show_about()
# Data Page
elif choose == "Data":
    data.show_data()
# Predictor Page
elif choose == "Predictor":
    predictor.show_predictor()
    pass
# Dashboard Page
elif choose == "Dash board":
    dashboard.show_dashboard()
