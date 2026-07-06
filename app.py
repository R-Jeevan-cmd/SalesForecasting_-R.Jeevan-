import streamlit as st # type: ignore

st.set_page_config(
    page_title="Retail Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Retail Sales Forecasting Dashboard")

st.markdown("""
Welcome to the **Retail Sales Forecasting Dashboard**.

This project includes:

- 📊 Sales Overview
- 📈 Sales Forecasting
- 🚨 Anomaly Detection
- 📦 Product Demand Segmentation

Use the navigation panel on the left to explore each page.
""")

st.info("Developed using Streamlit, Prophet, XGBoost, and Scikit-learn.")