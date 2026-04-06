import streamlit as st
import pandas as pd
import numpy as np

def apply_mitigation(df, target, protected_cols, audit_results):
    st.markdown("### 🛠️ Mitigation Engine")
    
    if not audit_results:
        st.info("Please run a Detection audit first.")
        return
    
    mitigated_df = df.copy()
    
    # Binary Target enforcement for math (1=Success, 0=Failure)
    if mitigated_df[target].dtype == 'object' or mitigated_df[target].nunique() > 2:
        mitigated_df['target_bin'] = pd.factorize(mitigated_df[target])[0]
    else:
        mitigated_df['target_bin'] = mitigated_df[target].astype(int)
        
    original_gap = audit_results.get('gap', 0)
    is_clf = audit_results.get('is_classification', True)
    
    if st.button("✨ Apply Intersectional Reweighing"):
        # Create unique key for each intersectional identity
        mitigated_df['group_key'] = mitigated_df[protected_cols].astype(str).agg(' & '.join, axis=1)
        
        n = len(mitigated_df)
        p_group = mitigated_df['group_key'].value_counts() / n
        p_outcome = mitigated_df['target_bin'].value_counts() / n
        
        # Calculate Four-Quadrant Weights (Kamiran-Calders)
        weights = []
        for _, row in mitigated_df.iterrows():
            g, y = row['group_key'], row['target_bin']
            p_gy = len(mitigated_df[(mitigated_df['group_key'] == g) & (mitigated_df['target_bin'] == y)]) / n
            # The Weight Formula: [P(Group) * P(Outcome)] / P(Group, Outcome)
            w = (p_group[g] * p_outcome[y]) / (p_gy + 1e-6)
            weights.append(w)
            
        mitigated_df['sample_weight'] = weights

        # Calculate Projected Bias using weighted averages
        def get_weighted_mean(group):
            return np.average(group['target_bin'], weights=group['sample_weight'])

        new_stats = mitigated_df.groupby('group_key').apply(get_weighted_mean)
        
        if is_clf:
            projected_gap = (new_stats.max() - new_stats.min()) * 100
            unit = "%"
        else:
            projected_gap = new_stats.max() / (new_stats.min() + 1e-6)
            unit = "x"

        # Results Display
        st.success("✅ Fairness Optimization Complete!")
        c1, c2 = st.columns(2)
        c1.metric("Original Bias", f"{original_gap:.2f}{unit}")
        c2.metric("Projected Bias", f"{projected_gap:.2f}{unit}", 
                  delta=f"-{original_gap - projected_gap:.1f}{unit}", delta_color="normal")
        
        st.info("💡 Data weights have been adjusted to ensure every group has the same average success rate.")
        
        # Exporting cleaned data
        export_df = mitigated_df.drop(columns=['group_key', 'target_bin'])
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Fair Dataset", csv, file_name="fair_dataset.csv", mime="text/csv")