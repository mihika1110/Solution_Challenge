import streamlit as st
import pandas as pd

import data_handler as m1    

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="FairFrame | AI Bias Auditor",
    page_icon="⚖️",
    layout="wide"
)

# 2. SIDEBAR NAVIGATION
st.sidebar.title("🛡️ FairFrame Control")
st.sidebar.info("Upload your dataset to begin the AI Ethics Audit.")
menu = st.sidebar.radio("Navigate", ["Dashboard", "Technical Docs", "About Team"])

# 3. MAIN APP LOGIC
if menu == "Dashboard":
    st.title("⚖️ FairFrame: Responsible AI Auditor")
    st.markdown("---")
    df, target_col, protected_col = m1.show_data_ui()

    if df is not None:
        st.success(f"Successfully loaded data. Auditing '{target_col}' based on '{protected_col}'.")
        
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("📊 Fairness Metrics")
            st.info("Member 2's charts and math will appear here.")
           
        with col_right:
            st.subheader("🤖 AI Auditor Insight")
            st.info("Member 3's Gemini explanation will appear here.")
           
elif menu == "Technical Docs":
    st.header("How FairFrame Works")
    st.write("This tool uses Fairlearn and Gemini 1.5 Flash to detect and explain AI bias.")

else:
    st.header("The Team")
    st.write("Built by a team of 4 for the Hackathon.")

