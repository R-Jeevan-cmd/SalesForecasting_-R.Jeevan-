import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

st.set_page_config(page_title="Forecast Explorer", layout="wide")

st.title("📈 Forecast Explorer")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    return df

df = load_data()

# -----------------------------
# Sidebar
# -----------------------------
forecast_type = st.sidebar.selectbox(
    "Forecast By",
    ["Category", "Region"]
)

if forecast_type == "Category":
    option = st.sidebar.selectbox(
        "Select Category",
        df["Category"].unique()
    )
    filtered = df[df["Category"] == option]

else:
    option = st.sidebar.selectbox(
        "Select Region",
        df["Region"].unique()
    )
    filtered = df[df["Region"] == option]

months = st.sidebar.slider(
    "Forecast Horizon (Months)",
    1,
    3,
    3
)

# -----------------------------
# Monthly Sales
# -----------------------------
monthly = (
    filtered
    .groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
    .sum()
    .reset_index()
)

monthly.columns = ["ds", "y"]

# -----------------------------
# Train Prophet
# -----------------------------
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)

model.fit(monthly)

future = model.make_future_dataframe(
    periods=months,
    freq="ME"
)

forecast = model.predict(future)

# -----------------------------
# Forecast Chart
# -----------------------------
st.subheader(f"{forecast_type}: {option}")

fig = model.plot(forecast)

st.pyplot(fig)

# -----------------------------
# Forecast Values
# -----------------------------
st.subheader("Forecast")

st.dataframe(
    forecast[
        ["ds","yhat","yhat_lower","yhat_upper"]
    ].tail(months)
)

# -----------------------------
# Model Accuracy
# -----------------------------
actual = monthly["y"][-3:]

predicted = forecast["yhat"][-(months+3):-months]

if len(actual) == len(predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    st.metric("MAE", round(mae,2))

    st.metric("RMSE", round(rmse,2))