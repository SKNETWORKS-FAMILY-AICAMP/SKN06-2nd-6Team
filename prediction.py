import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import io

#Home Page
# Provides an overview of the app and its purpose.

# Data Page
# Displays basic information about the dataset.
# Shows summary statistics of numerical variables.
# Provides the first few rows of the dataset.
# Conducts univariate and bivariate analysis.
# Presents additional analysis using pandas styling.


# Predictor Page
# Batch Prediction: Upload a CSV dataset containing customer information to predict churn.
# Online Prediction: Input customer details interactively to predict churn.

# Dashboard Page
# Provides visualizations and analytics related to customer churn.
# Includes research questions and key performance indicators.
# Offers insights through various charts and plots.

# History Page
# Tracks user interactions with the app.
# Displays a history log of actions performed by the user.
# Allows navigating back to previous points in history.



#사이드 바 생성
with st.sidebar:
    choose = option_menu("Churn prediction", ["About", "Data", "Predictor", "Dash borad"],
                         icons=['bi bi-cursor', 'bi bi-archive','bi bi-graph-up-arrow', 'bi bi-bar-chart'],
                         menu_icon="house", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "18px"}, 
        "nav-link": {"font-size": "13px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )




######################################################