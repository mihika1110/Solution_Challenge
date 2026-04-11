import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_data_ui():
    st.markdown("### 📥 Step 1: Data Ingestion & Auto-Cleaning")
    # Added info for Ingestion
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

        # --- 4. FEATURE SELECTION ---
        st.markdown("### Step 2: 🎯 Audit Configuration") 
        # Added info for Configuration
        st.info("Select the 'Target' (the outcome the AI predicts) and the 'Sensitive Attributes' (groups like race or sex) to check for unfair treatment.")
        
        col1, col2 = st.columns(2)

        with col1:
            target_col = st.selectbox(
                "Select the Goal Column (Target)", 
                options=df.columns,
                index=len(df.columns)-1
            )

        with col2:
            all_potential_features = [c for c in df.columns if c != target_col]
            protected_cols = st.multiselect(
                "Select Sensitive Attribute(s) for Deep Audit:", 
                options=all_potential_features,
                default=[all_potential_features[0]] if all_potential_features else []
            )

        # --- 5. DYNAMIC GROUP BALANCE CHART ---
        if protected_cols:
            st.write("#### ⚖️ Group Balance Audit")
            # Added info for Balance Audit
            st.info("Identify under-represented groups. Imbalanced data is often the primary source of algorithmic bias.")
            
            chart_cols = st.columns(len(protected_cols))
            for i, col_name in enumerate(protected_cols):
                with chart_cols[i]:
                    st.caption(f"Distribution: {col_name}")
                    balance = df[col_name].value_counts()
                    st.bar_chart(balance)

        # --- 6. PROXY DETECTOR (CORRELATION MATRIX) ---
        st.write("#### 🕵️ Proxy Detector")
        # Added info for Proxy Detector
        st.info("Scan for 'Proxies'—seemingly innocent features (like zip codes) that highly correlate with sensitive groups and hide bias.")
        
        if st.checkbox("Show Hidden Correlations (Search for Proxies)"):
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                # Ensure matplotlib is imported to avoid NameError
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("No numeric data available for correlation mapping.")
                
        return df, target_col, protected_cols, None
    
    return None, None, None, None
