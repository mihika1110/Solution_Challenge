import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def show_proxy_warning(df, protected_col):
    numeric_df = df.select_dtypes(include=['number'])

    if protected_col in df.columns and not numeric_df.empty:
        temp_df = df.copy()
        temp_df[protected_col] = temp_df[protected_col].astype('category').cat.codes

        corr = temp_df.corr(numeric_only=True)[protected_col].abs().sort_values(ascending=False)
        proxies = corr[1:3]

        if not proxies.empty and proxies.iloc[0] > 0.5:
            st.warning(
                f"⚠️ Proxy Detected: **{proxies.index[0]}** is highly correlated ({proxies.iloc[0]:.2f}) with "
                f"**{protected_col}**."
            )


def get_task_info(df, target):
    return df[target].nunique() <= 5


# ✅ NEW: Safe numeric conversion
def safe_convert_target(df, target):
    df = df.copy()

    # Try numeric conversion first
    df[target] = pd.to_numeric(df[target], errors='coerce')

    # If too many NaNs → treat as categorical
    if df[target].isna().sum() > 0.3 * len(df):
        df[target] = pd.factorize(df[target].astype(str))[0]

    return df


def run_audit(df, target, protected_cols, model=None):
    st.markdown("### 🔍 Intersectional Deep Dive")

    is_classification = get_task_info(df, target)

    df_temp = safe_convert_target(df, target)

    # Drop rows where target is still NaN
    df_temp = df_temp.dropna(subset=[target])

    if df_temp.empty:
        st.error("❌ Target column could not be converted to numeric. Please check your data.")
        return {}

    # Create groups
    df_temp['Demographic Groups'] = df_temp[protected_cols].astype(str).agg(' & '.join, axis=1)

    try:
        group_stats = (
            df_temp.groupby('Demographic Groups')[target]
            .mean()
            .dropna()
            .sort_values()
        )

        if group_stats.empty:
            st.warning("⚠️ Not enough valid data to compute statistics.")
            return {}

    except Exception as e:
        st.error(f"❌ Failed to compute group statistics: {e}")
        return {}

    # 📊 Plot
    st.write("#### Success Rate by Intersectional Group")

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [
        '#ff4b4b' if (x == group_stats.min() or x == group_stats.max()) else '#0078ff'
        for x in group_stats
    ]

    group_stats.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel("Mean Outcome")

    st.pyplot(fig)
    fig.savefig("bias_plot.png", bbox_inches='tight')

    # 📉 Bias Calculation
    if is_classification:
        gap = (group_stats.max() - group_stats.min()) * 100
        risk_level = "🔴 HIGH RISK" if gap > 15 else "🟡 MODERATE" if gap > 5 else "🟢 LOW RISK"
        display_gap = f"{gap:.2f}%"
    else:
        gap = group_stats.max() / (group_stats.min() + 1e-6)
        risk_level = "🔴 HIGH RISK" if gap > 1.5 else "🟡 MODERATE" if gap > 1.2 else "🟢 LOW RISK"
        display_gap = f"{gap:.2f}x"

    col1, col2 = st.columns(2)
    col1.metric("Bias Score", display_gap)
    col2.metric("Risk Status", risk_level)

    return {
        "gap": gap,
        "protected_cols": protected_cols,
        "is_classification": is_classification,
        "stats": group_stats,
        "target": target,
        "risk": risk_level
    }


def run_audit_all(df, target, all_cols):
    st.markdown("#### 📊 Global Attribute Risk Scan")

    temp_df = safe_convert_target(df, target)
    temp_df = temp_df.dropna(subset=[target])

    if temp_df.empty:
        st.warning("⚠️ Cannot perform global scan due to invalid target column.")
        return

    scan_results = []
    is_classification = get_task_info(df, target)

    for col in all_cols:
        if temp_df[col].nunique() > 20 or temp_df[col].nunique() < 2:
            continue

        try:
            group_stats = temp_df.groupby(col)[target].mean().dropna()

            if group_stats.empty:
                continue

            if is_classification:
                score = (group_stats.max() - group_stats.min()) * 100
                risk = "🔴 High" if score > 15 else "🟡 Med" if score > 5 else "🟢 Low"
            else:
                score = group_stats.max() / (group_stats.min() + 1e-6)
                risk = "🔴 High" if score > 1.5 else "🟡 Med" if score > 1.2 else "🟢 Low"

            scan_results.append({
                "Attribute": col,
                "Fairness Score": round(abs(score), 2),
                "Risk Level": risk
            })

        except Exception:
            continue

    if scan_results:
        results_df = pd.DataFrame(scan_results).sort_values(by="Fairness Score", ascending=False)
        st.table(results_df)
    else:
        st.info("No valid columns found for scanning.")