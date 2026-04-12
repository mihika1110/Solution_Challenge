import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ CACHE (prevents recomputation)
@st.cache_data
def compute_correlation(df):
    numeric_df = df.select_dtypes(include=['number'])
    return numeric_df.corr()


def show_data_ui():
    st.markdown("### 📥 Step 1: Data Ingestion & Auto-Cleaning")
    st.info("Drop your raw dataset here. The system will automatically remove empty features and technical noise to prepare for bias detection.")
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # 1. REMOVE EMPTY/NaN FEATURES
        threshold = len(df) * 0.5
        df = df.dropna(thresh=threshold, axis=1)
        df = df.dropna()
        
        # Remove technical columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed|^id$|index', case=False)]

        st.write(f"✅ Cleaned Dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
        
        st.write("#### Data Preview")
        st.dataframe(df.head(5), use_container_width=True)

        # --- FEATURE SELECTION ---
        st.markdown("### Step 2: 🎯 Audit Configuration") 
        st.info("Select the 'Target' and 'Sensitive Attributes' to audit fairness.")
        
        col1, col2 = st.columns(2)

        with col1:
            target_col = st.selectbox(
                "Select the Goal Column (Target)", 
                options=df.columns,
                index=len(df.columns)-1
            )

        with col2:
            all_potential_features = [c for c in df.columns if c != target_col]
            valid_protected_cols = [
            c for c in all_potential_features
            if df[c].nunique() < 20   # threshold
        ]

        if not valid_protected_cols:
            st.warning("⚠️ No suitable categorical columns found for fairness analysis.")
            protected_cols = []
        else:
            protected_cols = st.multiselect(
                "Select Sensitive Attribute(s):",
                options=valid_protected_cols,
                default=[valid_protected_cols[0]]
            )

        # --- GROUP BALANCE ---
        if protected_cols:
            st.write("#### ⚖️ Group Balance Audit")
            st.info("Identify under-represented groups.")
            
            chart_cols = st.columns(len(protected_cols))
            for i, col_name in enumerate(protected_cols):
                with chart_cols[i]:
                    st.caption(f"Distribution: {col_name}")
                    balance = df[col_name].value_counts()
                    st.bar_chart(balance)

        # --- PROXY DETECTOR ---
        st.write("#### 🕵️ Proxy Detector")
        st.info("Scan for features highly correlated with sensitive attributes.")

        if st.checkbox("Show Hidden Correlations (Search for Proxies)"):

            with st.spinner("🔍 Computing correlations..."):
                corr = compute_correlation(df)

                if corr.empty:
                    st.warning("No numeric data available for correlation mapping.")
                else:
                    # ✅ LIMIT SIZE (performance boost)
                    if corr.shape[0] > 20:
                        st.warning("Too many features — showing top 20 for performance.")
                        corr = corr.iloc[:20, :20]

                    fig, ax = plt.subplots(figsize=(10, 5))

                    # ❌ REMOVED annot=True (major speed fix)
                    sns.heatmap(
                        corr,
                        annot=False,
                        cmap="coolwarm",
                        ax=ax
                    )

                    st.pyplot(fig)

        return df, target_col, protected_cols, None
    
    return None, None, None, None