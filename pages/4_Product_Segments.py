import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Product Demand Segments", layout="wide")

st.title("📦 Product Demand Segmentation")

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Year"] = df["Order Date"].dt.year
    return df

df = load_data()

# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------

sales_volume = df.groupby("Sub-Category")["Sales"].sum()

avg_order = df.groupby("Sub-Category")["Sales"].mean()

monthly = (
    df.groupby(
        [
            "Sub-Category",
            pd.Grouper(key="Order Date", freq="ME")
        ]
    )["Sales"]
    .sum()
    .reset_index()
)

volatility = (
    monthly.groupby("Sub-Category")["Sales"]
    .std()
)

yearly = (
    df.groupby(["Sub-Category","Year"])["Sales"]
    .sum()
    .reset_index()
)

growth = yearly.pivot(
    index="Sub-Category",
    columns="Year",
    values="Sales"
)

growth_rate = (
    (growth.iloc[:,-1]-growth.iloc[:,0])
    /growth.iloc[:,0]
)*100

cluster_df = pd.DataFrame({

    "SalesVolume": sales_volume,

    "GrowthRate": growth_rate,

    "Volatility": volatility,

    "AverageOrder": avg_order

}).fillna(0)

# ---------------------------------------------------
# Scale
# ---------------------------------------------------

scaler = StandardScaler()

scaled = scaler.fit_transform(cluster_df)

# ---------------------------------------------------
# KMeans
# ---------------------------------------------------

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

cluster_df["Cluster"] = kmeans.fit_predict(scaled)

# ---------------------------------------------------
# Cluster Labels
# ---------------------------------------------------

cluster_labels = {
    0:"High Volume, Stable Demand",
    1:"Low Volume, High Volatility",
    2:"Growing Demand",
    3:"Declining Demand"
}

cluster_df["Demand Segment"] = (
    cluster_df["Cluster"]
    .map(cluster_labels)
)

# ---------------------------------------------------
# PCA
# ---------------------------------------------------

pca = PCA(n_components=2)

components = pca.fit_transform(scaled)

cluster_df["PC1"] = components[:,0]
cluster_df["PC2"] = components[:,1]

# ---------------------------------------------------
# Plot
# ---------------------------------------------------

st.subheader("Demand Clusters")

fig, ax = plt.subplots(figsize=(10,6))

scatter = ax.scatter(
    cluster_df["PC1"],
    cluster_df["PC2"],
    c=cluster_df["Cluster"],
    s=120
)

for i in range(len(cluster_df)):
    ax.text(
        cluster_df["PC1"].iloc[i],
        cluster_df["PC2"].iloc[i],
        cluster_df.index[i],
        fontsize=8
    )

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")

st.pyplot(fig)

# ---------------------------------------------------
# Table
# ---------------------------------------------------

st.subheader("Product Segments")

st.dataframe(
    cluster_df[
        [
            "SalesVolume",
            "GrowthRate",
            "Volatility",
            "AverageOrder",
            "Demand Segment"
        ]
    ]
)