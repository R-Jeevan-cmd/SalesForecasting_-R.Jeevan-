import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Anomaly Report", layout="wide")

st.title("🚨 Sales Anomaly Report")

# -----------------------------------
# Load Data
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    return df

df = load_data()

# -----------------------------------
# Weekly Sales
# -----------------------------------
weekly = (
    df.groupby(pd.Grouper(key="Order Date", freq="W"))["Sales"]
      .sum()
      .reset_index()
)

# -----------------------------------
# Isolation Forest
# -----------------------------------
iso = IsolationForest(
    contamination=0.05,
    random_state=42
)

weekly["Anomaly"] = iso.fit_predict(weekly[["Sales"]])

weekly["Anomaly"] = weekly["Anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})

# -----------------------------------
# Plot
# -----------------------------------
st.subheader("Weekly Sales with Detected Anomalies")

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(
    weekly["Order Date"],
    weekly["Sales"],
    label="Weekly Sales"
)

anomaly = weekly[weekly["Anomaly"] == "Anomaly"]

ax.scatter(
    anomaly["Order Date"],
    anomaly["Sales"],
    color="red",
    s=80,
    label="Anomaly"
)

ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()

st.pyplot(fig)

# -----------------------------------
# Table
# -----------------------------------
st.subheader("Detected Anomalies")

st.dataframe(
    anomaly[
        ["Order Date", "Sales"]
    ].reset_index(drop=True)
)
