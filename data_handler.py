import streamlit as st
import pandas as pd

def show_data_ui():
    st.markdown("### 📥 Step 1: Data Ingestion & Auto-Cleaning")
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

        st.divider()
        target_col = st.selectbox("Select the Goal Column (Target)", options=df.columns)

        # FIX: Removed m2_audit.run_audit_all call from here 
        # to stop it from printing twice (it's already in app.py)

        st.divider()
        all_potential_features = [c for c in df.columns if c != target_col]
        protected_cols = st.multiselect(
            "Select Sensitive Attribute(s) for Deep Audit:", 
            options=all_potential_features,
            default=[all_potential_features[0]] if all_potential_features else []
        )

        return df, target_col, protected_cols, None
    
    return None, None, None, None