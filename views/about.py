import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_about():
    st.header("📑 About Our Services")
    
    # Waze
    st.markdown("### ✔ What is 'Waze'?") 
    image_url = "https://lh3.googleusercontent.com/AuI79N0xGtd1W7ARQrlr_1ktLgmamXdTw6bcQjqvoupKbuVSNjX4LMhztGUJbqCfKcnB65n3CD3CTwfPYAfSpsdSdS4YUtCHrmgZNw=h630-w1200"
    st.image(image_url, caption="Waze", width=500)
    
    st.markdown(
        """
        <p style='font-size: 25px; line-height: 1.6;'>
            <li><b>Top-Ranked navigation app in the U.S.</b></li>
            <li><b>Real-Time, User-Driven Insights</b></li>
            <li><b>Personalized Navigation Experience</b></li>
        </p>
        
        <p style='font-size: 16px; line-height: 1.6;'>
        Waze is a community-driven navigation app that provides real-time traffic information, route suggestions, and alerts based on crowd-sourced data. 
        With millions of users contributing live updates, Waze helps drivers find the fastest routes, avoid traffic jams, and stay informed about road conditions.
        </p>
        
        <p style='font-size: 16px; line-height: 1.6;'>
        Learn more about Waze: <a href='https://www.waze.com/about' target='_blank'>https://www.waze.com/about</a>
        </p>
        """, 
        unsafe_allow_html=True
    )
    st.divider() 
    
    # Objective
    st.markdown("### ✔ Objective") 
    st.markdown(
        """
        <br>
        <p style='font-size: 25px; line-height: 1.6;'>
        <b>Predict your potential churn customers and build the best strategy!</b></p> <br><br>
        <p style='font-size: 16px; line-height: 1.6;'>
        We are providing a customer churn rate prediction service for <b>Waze</b>, 
        one of the most popular mobile navigation applications, to help you find valuable business insights.
        </p>
        """, unsafe_allow_html=True
    )
    st.divider() 
    
    # Guide for Each Category
    st.markdown("### ✔ Guide for Each Category")
    st.markdown(
        """
        <ul style='font-size: 16px; line-height: 1.6;'>
            <li><b>Data</b>: Provides an overview of the data used for predictions.</li>
            <li><b>Predictor</b>: Make predictions on customer churn likelihood.</li>
            <li><b>Dashboard</b>: Visualize insights and trends to support your strategies.</li>
        </ul>
        """, unsafe_allow_html=True
    )