import streamlit as st
from risk_assessment import risk_assessment_page

def main():
    st.set_page_config(
        page_title="Breast Cancer Recurrence Prediction",
        page_icon="🎗️",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    risk_assessment_page()

if __name__ == "__main__":
    main()
