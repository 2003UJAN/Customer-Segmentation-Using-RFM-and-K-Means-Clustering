import streamlit as st
import pandas as pd
from rfm_segmentation import load_data, compute_rfm, preprocess_rfm, apply_kmeans, reduce_pca
from utils import plot_segments

st.set_page_config(page_title="Adidas Customer Segmentation", layout="wide")

st.title("🎯 Adidas Customer Segmentation using RFM and K-Means")

uploaded = st.file_uploader("Upload Online Retail Dataset CSV", type="csv")
if uploaded:
    df = load_data(uploaded)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalSum'] = df['Quantity'] * df['UnitPrice']

    rfm = compute_rfm(df)
    st.subheader("🔍 RFM Table Sample")
    st.dataframe(rfm.head())

    scaled_rfm = preprocess_rfm(rfm)
    clusters = apply_kmeans(scaled_rfm, n_clusters=4)
    rfm['Cluster'] = clusters

    components = reduce_pca(scaled_rfm)
    fig = plot_segments(components, clusters)
    st.plotly_chart(fig)

    st.subheader("📊 Segment Counts")
    st.bar_chart(rfm['Cluster'].value_counts())

    st.subheader("💡 Segment Definitions (Example)")
    st.markdown("""
    - **0:** Loyal Customers  
    - **1:** At Risk  
    - **2:** Potential Loyalist  
    - **3:** New Customers  
    """)

else:
    st.info("Please upload the dataset to begin.")
