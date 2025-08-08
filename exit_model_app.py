import streamlit as st
import pandas as pd
import numpy as np

def render_exit_model():
    """
    Renders the interactive exit probability model.
    """
    st.title("Exit Probability Model")

    st.write("This tool calculates the probability of a startup achieving a certain exit valuation based on its funding stage and the number of funding rounds raised.")

    # --- Data Input ---
    st.header("1. Input Probabilities")

    col1, col2 = st.columns(2)

    with col1:
        # Exit Valuation vs. Highest Fundraising Stage
        st.subheader("Exit Valuation vs. Highest Fundraising Stage (%)")
        exit_val_data = {
            'Pre-Seed': [80, 20, 0, 0, 0, 0],
            'Seed': [50, 20, 20, 10, 0, 0],
            'A': [10, 15, 35, 20, 10, 10],
            'B': [0, 10, 30, 30, 15, 15]
        }
        exit_val_index = ['0 (Failure)', '10M', '50M', '100M', '500M', '1B']
        exit_val_df = pd.DataFrame(exit_val_data, index=exit_val_index)
        
        edited_exit_val_df = st.data_editor(exit_val_df, key="exit_val_editor", height=250)

    with col2:
        # Total Rounds vs. Entry Stage
        st.subheader("Total Rounds vs. Entry Stage (%)")
        rounds_data = {
            'Pre-Seed': [33, 33, 33],
            'Seed': [50, 25, 25]
        }
        rounds_index = [1, 2, 3]
        rounds_df = pd.DataFrame(rounds_data, index=rounds_index)
        
        edited_rounds_df = st.data_editor(rounds_df, key="rounds_editor", height=150)

    # Convert percentages to probabilities
    exit_val_prob = edited_exit_val_df / 100.0
    rounds_prob = edited_rounds_df / 100.0

    # --- Scenario Parameters ---
    st.header("2. Scenario Parameters")
    st.write("Adjust the failure rate for custom scenarios based on the 'Pre-Seed Entry' benchmark.")

    col_param1, col_param2 = st.columns(2)
    with col_param1:
        large_pre_seed_fail_reduction = st.slider(
            "Large Pre-Seed: Reduction in failure rate (%)",
            min_value=0,
            max_value=100,
            value=20,
            help="How much less likely a 'Large Pre-seed' startup is to fail (exit at 0) compared to the baseline Pre-seed."
        )

    with col_param2:
        exploratory_checks_fail_increase = st.slider(
            "Exploratory Checks: Increase in failure rate (%)",
            min_value=0,
            max_value=200,
            value=30,
            help="How much more likely an 'Exploratory Checks' startup is to fail (exit at 0) compared to the baseline Pre-seed."
        )

    # --- Calculation ---
    st.header("3. Calculated Exit Probabilities")

    # Pre-Seed Calculation (Benchmark)
    pre_seed_exit_prob = pd.Series(0.0, index=exit_val_prob.index)
    for i, stage in enumerate(['Pre-Seed', 'Seed', 'A']):
        if i < len(rounds_prob['Pre-Seed']):
            pre_seed_exit_prob += rounds_prob['Pre-Seed'].iloc[i] * exit_val_prob[stage]

    # Seed Calculation
    seed_exit_prob = pd.Series(0.0, index=exit_val_prob.index)
    for i, stage in enumerate(['Seed', 'A', 'B']):
        if i < len(rounds_prob['Seed']):
            seed_exit_prob += rounds_prob['Seed'].iloc[i] * exit_val_prob[stage]

    # --- Large Pre-seed Calculation ---
    large_pre_seed_prob = pre_seed_exit_prob.copy()
    original_failure_prob = large_pre_seed_prob.iloc[0]
    original_success_total = 1 - original_failure_prob

    reduction_amount = original_failure_prob * (large_pre_seed_fail_reduction / 100.0)
    new_failure_prob = original_failure_prob - reduction_amount
    large_pre_seed_prob.iloc[0] = new_failure_prob
    
    new_success_total = original_success_total + reduction_amount

    if original_success_total > 0:
        scaling_factor = new_success_total / original_success_total
        large_pre_seed_prob.iloc[1:] = pre_seed_exit_prob.iloc[1:] * scaling_factor
    elif new_success_total > 0 and len(large_pre_seed_prob.iloc[1:]) > 0:
        large_pre_seed_prob.iloc[1:] = new_success_total / len(large_pre_seed_prob.iloc[1:])

    # --- Exploratory Checks Calculation ---
    exploratory_checks_prob = pre_seed_exit_prob.copy()
    original_failure_prob_exp = exploratory_checks_prob.iloc[0]
    original_success_total_exp = 1 - original_failure_prob_exp

    increase_factor = 1 + (exploratory_checks_fail_increase / 100.0)
    desired_new_failure_prob = original_failure_prob_exp * increase_factor
    
    new_success_total_exp = max(0, 1 - desired_new_failure_prob)
    
    new_failure_prob_exp = 1 - new_success_total_exp
    exploratory_checks_prob.iloc[0] = new_failure_prob_exp

    if original_success_total_exp > 0:
        scaling_factor_exp = new_success_total_exp / original_success_total_exp
        exploratory_checks_prob.iloc[1:] = pre_seed_exit_prob.iloc[1:] * scaling_factor_exp
    else:
        exploratory_checks_prob.iloc[1:] = 0

    # --- Display Results ---
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)

    with res_col1:
        st.subheader("Pre-Seed Entry")
        st.dataframe(pre_seed_exit_prob.apply(lambda x: f"{x:.2%}"), use_container_width=True)

    with res_col2:
        st.subheader("Seed Entry")
        st.dataframe(seed_exit_prob.apply(lambda x: f"{x:.2%}"), use_container_width=True)

    with res_col3:
        st.subheader("Large Pre-seed")
        st.dataframe(large_pre_seed_prob.apply(lambda x: f"{x:.2%}"), use_container_width=True)

    with res_col4:
        st.subheader("Exploratory Checks")
        st.dataframe(exploratory_checks_prob.apply(lambda x: f"{x:.2%}"), use_container_width=True)
        
    # --- Integration with VC Fund Model ---
    st.header("4. Use in VC Fund Model")
    if st.button("Use Probabilities in Fund Model"):
        st.session_state['exit_model_probabilities'] = {
            "Pre-Seed Entry": pre_seed_exit_prob.to_dict(),
            "Seed Entry": seed_exit_prob.to_dict(),
            "Large Pre-seed": large_pre_seed_prob.to_dict()
        }
        st.success("Probabilities have been saved and are ready to be applied in the 'VC Fund Model' tab.")
