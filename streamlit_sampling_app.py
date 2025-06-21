
import streamlit as st
import pandas as pd
import numpy as np
import random

st.set_page_config(page_title="Sampling Tool", layout="centered")

st.title("📊 Sampling Tool")

HELP_TEXT = """### 📌 Sampling Methods Explained

1. **Simple Random Sampling**  
   Each member of the population has an equal chance of being selected.

2. **Systematic Sampling**  
   Every k-th item is selected from a randomly chosen starting point.

3. **Stratified Sampling**  
   The population is divided into strata (groups), and random samples are taken from each stratum.

4. **Cluster Sampling**  
   The population is divided into clusters, and a random selection of entire clusters is sampled.

5. **PPS Sampling**  
   Probability Proportional to Size: units are selected based on size or weight measures.
"""

def simple_sample(df, n):
    return df.sample(n=n, random_state=42).reset_index(drop=True)

def stratified_sample(df, n, col):
    grp = df.groupby(col)
    return grp.apply(lambda x: x.sample(n=max(1,int(round(n*len(x)/len(df)))), random_state=42)).reset_index(drop=True)

def cluster_sample(df, n, col):
    clusters = df[col].dropna().unique()
    sel = random.sample(list(clusters), min(len(clusters), n))
    return df[df[col].isin(sel)].reset_index(drop=True)

def systematic_sample(df, n):
    step = len(df)//n
    start = random.randint(0, step-1)
    idx = list(range(start, len(df), step))[:n]
    return df.iloc[idx].reset_index(drop=True)

def pps_sample(df, n, col, auto=True, bins=None, labels=None, weights=None):
    if auto:
        df2 = df[df[col]>0].copy()
        df2['prob'] = df2[col]/df2[col].sum()
    else:
        df2 = df.copy()
        df2[col+'_bin'] = pd.qcut(df2[col], q=bins, labels=labels)
        m = dict(zip(labels, weights))
        df2['prob'] = df2[col+'_bin'].map(m)
        df2['prob'] /= df2['prob'].sum()
    return df2.sample(n=n, weights='prob', random_state=42).reset_index(drop=True)

uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File loaded successfully!")
    st.write("Preview of data:", df.head())

    method = st.selectbox("📌 Choose Sampling Method", ["Simple Random", "Systematic", "Stratified", "Cluster", "PPS", "Help"])

    if method == "Help":
        st.markdown(HELP_TEXT)
    else:
        n = st.number_input("🎯 Enter sample size", min_value=1, max_value=len(df), value=min(10, len(df)))

        if st.button("Run Sampling"):
            try:
                if method == "Simple Random":
                    result = simple_sample(df, n)

                elif method == "Systematic":
                    result = systematic_sample(df, n)

                elif method == "Stratified":
                    col = st.selectbox("Select column for stratification", df.columns)
                    result = stratified_sample(df, n, col)

                elif method == "Cluster":
                    col = st.selectbox("Select column for cluster sampling", df.columns)
                    result = cluster_sample(df, n, col)

                elif method == "PPS":
                    col = st.selectbox("Select weight column for PPS", df.columns)
                    auto = st.checkbox("Auto-calculate PPS weights", value=True)
                    if auto:
                        result = pps_sample(df, n, col, auto=True)
                    else:
                        bins = st.number_input("Enter number of bins", min_value=2, max_value=10, value=3)
                        label_text = st.text_input(f"Enter {bins} comma-separated labels")
                        weight_text = st.text_input(f"Enter {bins} comma-separated weights")
                        if label_text and weight_text:
                            labels = [l.strip() for l in label_text.split(",")]
                            weights = [float(w.strip()) for w in weight_text.split(",")]
                            result = pps_sample(df, n, col, auto=False, bins=bins, labels=labels, weights=weights)
                        else:
                            st.warning("Please enter both labels and weights.")

                st.success("✅ Sampling completed!")
                st.write(result)

                csv = result.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Sampled Data as CSV", data=csv, file_name="sampled_output.csv", mime='text/csv')
            except Exception as e:
                st.error(f"❌ Sampling failed: {e}")
else:
    st.info("👈 Upload a CSV file to get started.")
