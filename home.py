# Importo librerie utili
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
from itertools import product
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from pages.risk_assessment import risk_assessment_page

import pickle

##################################################################################################################################################################################################

def home_page():

    
    st.write("# 🎗️ Breast Cancer Recurrence Prediction")
    st.write("")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Welcome to the Breast Cancer Recurrence Prediction Tool

        This application helps doctors and patients understand the factors that influence breast cancer recurrence risk.

        With this tool, you can:
        - Upload patient data for analysis
        - Explore risk factors and their impact
        - Visualize prediction results
        - Interact with the model to see how changes in parameters affect outcomes

        #### How to use this tool:
        1. Navigate through the different sections using the sidebar menu
        2. Upload your data in the Data Analysis section
        3. Train the prediction model with your parameters
        4. Use the interactive assessment to explore different scenarios
        """)

    with col2:
        st.image("https://img.freepik.com/free-vector/breast-cancer-awareness-ribbon_23-2147877335.jpg", use_column_width=True)

    st.markdown("---")

    # Key statistics in metric cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Breast Cancer Cases Worldwide", value="2.3M+", delta="annually")
    with col2:
        st.metric(label="5-Year Survival Rate", value="90%", delta="+5% from 2000", delta_color="normal")
    with col3:
        st.metric(label="Recurrence Risk", value="20-30%", delta="varies by type", delta_color="off")

def main():
    st.set_page_config(page_title="Breast Cancer Recurrence Prediction", page_icon="🎗️", layout="wide")
    st.markdown('''<style> section.main > div {max-width:80rem} </style>''', unsafe_allow_html=True)

    pages = ["Home", "Risk Assessment"]

    page = st.sidebar.selectbox("Select a page", pages)

    st.sidebar.markdown("""
        <style>
            .sidebar .sidebar-content {
                padding: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    if page == "Home":
        home_page()
    elif page == "Risk Assessment":
        risk_assessment_page()

if __name__ == "__main__":
    main()  


