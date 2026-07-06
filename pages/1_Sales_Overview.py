import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sales Overview", layout="wide")

st.title("📊 Sales Overview Dashboard")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv")

    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("Filters")

regions = st.sidebar.multiselect(
    "Select Region",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)

categories = st.sidebar.multiselect(
    "Select Category",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

filtered_df = df[
    (df["Region"].isin(regions)) &
    (df["Category"].isin(categories))
]

# -----------------------------
# Total Sales by Year
# -----------------------------

st.subheader("Total Sales by Year")

year_sales = filtered_df.groupby("Year")["Sales"].sum()

fig, ax = plt.subplots(figsize=(8,4))
year_sales.plot(kind="bar", ax=ax)

ax.set_ylabel("Sales")
ax.set_xlabel("Year")

st.pyplot(fig)

# -----------------------------
# Monthly Sales Trend
# -----------------------------

st.subheader("Monthly Sales Trend")

monthly_sales = (
    filtered_df
    .groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
    .sum()
)

fig2, ax2 = plt.subplots(figsize=(10,4))

monthly_sales.plot(ax=ax2)

ax2.set_ylabel("Sales")
ax2.set_xlabel("Date")

st.pyplot(fig2)

# -----------------------------
# Data Preview
# -----------------------------

st.subheader("Filtered Dataset")

st.dataframe(filtered_df)