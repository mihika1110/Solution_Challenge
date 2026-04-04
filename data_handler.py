import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_data_ui():
  
    st.markdown("### 📥 Step 1: Data Ingestion & Quality Check")
    
    # 1. File Uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
      
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # 2. Data Cleaning 
        initial_count = len(df)
        df = df.dropna()
        final_count = len(df)
        
        if initial_count > final_count:
            st.warning(f"Removed {initial_count - final_count} rows with missing values for better accuracy.")

        # 3. Data Preview
        st.write("#### Data Preview")
        st.dataframe(df.head(5), use_container_width=True)

        # 4. Feature Selection 
        st.write("#### Configure Audit")
        col1, col2 = st.columns(2)
        
        with col1:
            target_col = st.selectbox(
                "Select the Goal Column (e.g., Loan_Status)", 
                options=df.columns,
                help="This is the outcome the AI is deciding."
            )
        with col2:
            protected_col = st.selectbox(
                "Select the Sensitive Group (e.g., Gender)", 
                options=df.columns,
                help="This is the attribute we want to check for fairness."
            )

        # 5. Data Balance Chart
        st.write("#### Group Balance Audit")
        
        balance = df[protected_col].value_counts().reset_index()
        balance.columns = [protected_col, 'Count'] 
        
        st.bar_chart(balance.set_index(protected_col))

        # 6. Correlation Matrix 
        if st.checkbox("Show Hidden Correlations (Proxy Detector)"):
            st.write("Checking if neutral columns are secretly linked to the protected group...")
            # Simple numeric correlation for demo
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                fig, ax = plt.subplots()
                sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)

        return df, target_col, protected_col
    
    return None, None, None