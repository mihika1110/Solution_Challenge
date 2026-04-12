import streamlit as st
import data_handler as m1 
import bias_detector as m2_audit 
import bias_fixer as m2_fix
import ai_auditor as ai 
import about_team as team  

st.set_page_config(page_title="FairFrame Pro | AI Auditor", page_icon="⚖️", layout="wide")

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

st.sidebar.button(
    "🌙 Toggle Dark Mode" if not st.session_state.dark_mode else "☀️ Toggle Light Mode", 
    on_click=toggle_theme
)

# --- THEME ---
if st.session_state.dark_mode:
    st.markdown("""
        <style>
            header[data-testid="stHeader"] { background-color: #0e1117 !important; }
            .stApp { background-color: #0e1117; color: #ffffff; }

            .stButton>button {
                background-color: #21262d !important;
                color: #ffffff !important;
                border: 1px solid #30363d !important;
                width: 100%;
            }

            [data-testid="stSidebar"] {
                background-color: #0d1117 !important;
            }

            h1, h2, h3, h4, h5, h6, p, label {
                color: #ffffff !important;
            }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            header[data-testid="stHeader"] { background-color: #ffffff !important; }
        </style>
    """, unsafe_allow_html=True)


# --- SIDEBAR ---
st.sidebar.title("FairFrame Control")
st.sidebar.info("Upload your dataset and model to begin the AI Ethics Audit.")
menu = st.sidebar.radio("Navigate", ["Audit Dashboard", "Technical Methodology", "About Team"])


# =========================
# 🚀 MAIN DASHBOARD
# =========================
if menu == "Audit Dashboard":
    st.title("Responsible AI Audit Dashboard")
    st.markdown("---")
    
    df, target_col, protected_cols, model = m1.show_data_ui()

    if df is not None and target_col and protected_cols:

        # --- STEP 3: BIAS DETECTION ---
        st.markdown("---")
        st.markdown("### Step 3: 📊 Bias Detection Analysis") 
        st.info("Scanning features for bias patterns and disparity scores.")
        
        all_pot = [c for c in df.columns if c != target_col]
        m2_audit.run_audit_all(df, target_col, all_pot)
        
        st.divider()
        st.session_state.results = m2_audit.run_audit(df, target_col, protected_cols)

        # 🤖 AI ANALYSIS
        with st.container(border=True):
            st.markdown("🤖 AI Analysis Insight")
            with st.spinner("🤖 AI thinking..."):
                insight = ai.generate_micro_insight(
                    "analysis",
                    stats=st.session_state.results
                )
                st.info(insight)

        # --- STEP 4: MITIGATION ---
        if "results" in st.session_state:
            st.markdown("---")
            st.markdown("### Step 4: 🛠️ Bias Mitigation Engine")
            st.info("Applying fairness corrections.")
            
            with st.container(border=True):
                m2_fix.apply_mitigation(df, target_col, protected_cols, st.session_state.results)

        # 🤖 AI MITIGATION
        if "results" in st.session_state:
            with st.container(border=True):
                st.markdown("🤖 AI Mitigation Strategy")
                with st.spinner("🤖 AI thinking..."):
                    insight = ai.generate_micro_insight("mitigation")
                    st.info(insight)

        # --- STEP 5: REPORT ---
        if "results" in st.session_state:
            st.markdown("---")
            st.markdown("### Step 5: 📜 Professional Audit Report")
            
            m2_audit.show_proxy_warning(df, protected_cols[0])

            with st.spinner("Generating professional report..."):
                insight = ai.generate_ai_report(st.session_state.results)

            with st.container(border=True):
                st.markdown("#### 🤖 AI Auditor Findings")
                st.markdown(insight["finding"])
                
                col_r1, col_r2 = st.columns([1, 3])
                with col_r1:
                    st.metric("Risk Level", insight["risk"])
                with col_r2:
                    st.success("Analysis complete.")

            pdf_data = ai.create_pdf(st.session_state.results, insight["finding"])
            st.download_button(
                label="📥 Download Professional Report (PDF)",
                data=pdf_data,
                file_name="FairFrame_Audit_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

            st.success("✅ Fairness-aware AI auditing completed.")


# =========================
# 📘 TECHNICAL PAGE (SAFE PLACEHOLDER)
# =========================
elif menu == "Technical Methodology":
    st.title("📘 Technical Methodology")
    st.info("This section will describe the fairness metrics and mitigation techniques used in the system.")


# =========================
# 👩‍💻 ABOUT TEAM
# =========================
elif menu == "About Team": 
    team.show_about_team()


# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.write("🚀 Powered by Gemini 1.5 Flash")