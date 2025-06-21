
import streamlit as st
import pandas as pd
import numpy as np
import random

st.set_page_config(page_title="Sampling Tool", layout="centered")
st.title("📊 Sampling Tool")

st.markdown("Upload a CSV file to get started.")
uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df.head())

    sample_mode = st.radio("Choose Sampling Mode", ["Manual", "Automatic"])

    method = st.selectbox("Select Sampling Method", [
        "Simple Random",
        "Systematic",
        "Stratified",
        "Cluster",
        "PPS",
        "All"
    ])

    n = st.number_input("Enter Sample Size", min_value=1, max_value=len(df), value=min(10, len(df)))

    def simple_sample(df, n):
        return df.sample(n=n, random_state=42).reset_index(drop=True)

    def systematic_sample(df, n):
        step = len(df) // n
        start = random.randint(0, step-1)
        return df.iloc[start::step][:n].reset_index(drop=True)

    def stratified_sample(df, n):
        if sample_mode == "Manual":
            col = st.text_input("Enter column for stratification")
        else:
            col = df.select_dtypes(include='object').columns[0] if len(df.select_dtypes(include='object').columns) else df.columns[0]
        if col not in df.columns:
            st.warning(f"Column '{col}' not found in data.")
            return pd.DataFrame()
        grp = df.groupby(col)
        return grp.apply(lambda x: x.sample(n=max(1,int(round(n*len(x)/len(df)))), random_state=42)).reset_index(drop=True)

    def cluster_sample(df, n):
        if sample_mode == "Manual":
            col = st.text_input("Enter column for cluster")
        else:
            col = df.select_dtypes(include='object').columns[0] if len(df.select_dtypes(include='object').columns) else df.columns[0]
        if col not in df.columns:
            st.warning(f"Column '{col}' not found in data.")
            return pd.DataFrame()
        clusters = df[col].dropna().unique()
        sel = random.sample(list(clusters), min(len(clusters), n))
        return df[df[col].isin(sel)].reset_index(drop=True)

    def pps_sample(df, n):
        if sample_mode == "Manual":
            pps_col = st.selectbox("Select PPS column", df.columns)
            mode = st.radio("Use automatic weights?", ["Yes", "No"])
        else:
            pps_col = df.select_dtypes(include=np.number).columns[0]
            mode = "Yes"

        if pps_col not in df.columns:
            st.warning("Invalid PPS column.")
            return pd.DataFrame()

        df2 = df.copy()
        if mode == "Yes":
            df2 = df2[df2[pps_col] > 0].copy()
            df2['prob'] = df2[pps_col] / df2[pps_col].sum()
        else:
            bins = st.number_input("Number of bins", min_value=2, max_value=10, value=3)
            labels = st.text_input("Enter labels (comma-separated)")
            weights = st.text_input("Enter weights (comma-separated)")
            try:
                lbls = [l.strip() for l in labels.split(',')]
                wts = [float(x.strip()) for x in weights.split(',')]
                if len(lbls) != bins or len(wts) != bins:
                    st.error("Label or weight count mismatch.")
                    return pd.DataFrame()
                df2['bin'] = pd.qcut(df2[pps_col], q=bins, labels=lbls)
                wmap = dict(zip(lbls, wts))
                df2['prob'] = df2['bin'].map(wmap)
                df2['prob'] = df2['prob'] / df2['prob'].sum()
            except:
                st.error("Error in binning weights.")
                return pd.DataFrame()
        try:
            return df2.sample(n=n, weights='prob', random_state=42).reset_index(drop=True)
        except:
            st.error("PPS sampling failed.")
            return pd.DataFrame()

    def preview(df_out, name):
        st.subheader(f"{name} Sample Preview")
        st.dataframe(df_out)
        csv = df_out.to_csv(index=False).encode('utf-8')
        st.download_button(f"Download {name} Sample", csv, file_name=f"{name}_sample.csv", mime='text/csv')

    if st.button("Run Sampling"):
        if method == "Simple Random":
            preview(simple_sample(df, n), method)
        elif method == "Systematic":
            preview(systematic_sample(df, n), method)
        elif method == "Stratified":
            strat_df = stratified_sample(df, n)
            if not strat_df.empty:
                preview(strat_df, method)
        elif method == "Cluster":
            clust_df = cluster_sample(df, n)
            if not clust_df.empty:
                preview(clust_df, method)
        elif method == "PPS":
            pps_df = pps_sample(df, n)
            if not pps_df.empty:
                preview(pps_df, method)
        elif method == "All":
            st.info("Running all sampling methods...")
            preview(simple_sample(df, n), "Simple Random")
            preview(systematic_sample(df, n), "Systematic")
            strat_df = stratified_sample(df, n)
            if not strat_df.empty:
                preview(strat_df, "Stratified")
            clust_df = cluster_sample(df, n)
            if not clust_df.empty:
                preview(clust_df, "Cluster")
            pps_df = pps_sample(df, n)
            if not pps_df.empty:
                preview(pps_df, "PPS")
