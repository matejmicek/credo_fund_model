import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go
from scipy.stats import beta
import json
from copy import deepcopy
import os

# Allow larger dataframes to be styled
pd.set_option("styler.render.max_elements", 1_000_000) # Set a large number instead of None

# --- Main Application Logic ---

def update_model_value(key_path, widget_key):
    """
    Generic callback to update a value in the nested fund_model dictionary.
    key_path is a list of keys to navigate the dictionary.
    """
    target = st.session_state.fund_model
    for key in key_path[:-1]:
        target = target[key]
    
    # For sliders that return a tuple (min, max), we need to handle them differently
    if isinstance(st.session_state[widget_key], tuple) and len(key_path[-1]) == 2:
        target[key_path[-1][0]], target[key_path[-1][1]] = st.session_state[widget_key]
    else:
        target[key_path[-1]] = st.session_state[widget_key]

    # After any model change, invalidate the previous simulation results
    if 'simulation_results' in st.session_state:
        del st.session_state.simulation_results

def add_scenario(bucket_key):
    """Callback to add a new, empty scenario to a bucket."""
    model = st.session_state.fund_model
    scenarios = model['buckets'][bucket_key].get('scenarios', [])
    scenarios.append({
        'name': f'New Scenario', 'probability': 0, 
        'exit_valuation_min': 10.0, 'exit_valuation_max': 20.0, 
        'exit_year_min': 5, 'exit_year_max': 8,
        'exit_dilution_pct': 20,
    })
    if 'simulation_results' in st.session_state:
        del st.session_state.simulation_results

def remove_scenario(bucket_key, scenario_index):
    """Callback to remove a scenario from a bucket."""
    model = st.session_state.fund_model
    if scenario_index < len(model['buckets'][bucket_key].get('scenarios', [])):
        del model['buckets'][bucket_key]['scenarios'][scenario_index]
    if 'simulation_results' in st.session_state:
        del st.session_state.simulation_results

def add_bucket():
    """Callback to add a new, empty bucket."""
    model = st.session_state.fund_model
    buckets = model.get('buckets', {})
    
    # Find the highest existing key to create a new unique key
    new_key = str(max([int(k) for k in buckets.keys()] + [-1]) + 1)
    
    buckets[new_key] = {
        'name': f'New Bucket {new_key}', 'percentage': 0,
        'deploy_y1': 100, 'deploy_y2': 0, 'deploy_y3': 0, 'deploy_y4': 0,
        'avg_ticket': 1.0,
        'entry_valuation_min': 5.0, 'entry_valuation_max': 10.0,
        'follow_on_allocation_pct': 0,
        'follow_on_probability': 50,
        'follow_on_timing': 2.0,
        'follow_on_size_pct_of_initial': 200,
        'follow_on_valuation_multiple': 2.0,
        'scenarios': [
            {'name': 'Default Scenario', 'probability': 100, 'exit_valuation_min': 10.0, 'exit_valuation_max': 20.0, 'exit_year_min': 5, 'exit_year_max': 8, 'exit_dilution_pct': 20},
        ]
    }
    model['buckets'] = buckets # Ensure the change is saved back
    if 'simulation_results' in st.session_state:
        del st.session_state.simulation_results

def remove_bucket(bucket_key):
    """Callback to remove a bucket by its key."""
    model = st.session_state.fund_model
    if bucket_key in model.get('buckets', {}):
        del model['buckets'][bucket_key]
    if 'simulation_results' in st.session_state:
        del st.session_state.simulation_results

def render_fund_model_ui():
    """Renders the main UI once the model data is loaded into session state."""
    st.title("🔮 Probabilistic VC Fund Model")

    # --- Sidebar for Global Fund Configuration ---
    st.sidebar.header("Global Fund Configuration")

    st.sidebar.download_button(
        label="Save Current Model as JSON",
        data=json.dumps(st.session_state.fund_model, indent=2),
        file_name='fund_model_config.json',
        mime='application/json',
        help="Save all the current model parameters to a JSON file."
    )
    
    st.sidebar.divider()

    model = st.session_state.fund_model

    st.sidebar.text_input(
        "Model Display Name",
        value=model.get('display_name', 'My Custom Model'),
        key='fm_display_name',
        on_change=update_model_value,
        args=(['display_name'], 'fm_display_name'),
        help="A name for this model, used for display purposes."
    )

    st.sidebar.number_input(
        "Fund Size ($ Millions)", min_value=1,
        value=model.get('fund_size', 100),
        key='fm_fund_size',
        on_change=update_model_value,
        args=(['fund_size'], 'fm_fund_size')
    )
    
    st.sidebar.slider(
        "Follow-on Capital Reserve (%)", 0, 80,
        value=model.get('follow_on_reserve', 40), step=5,
        key='fm_follow_on_reserve',
        on_change=update_model_value,
        args=(['follow_on_reserve'], 'fm_follow_on_reserve'),
        help="Percentage of the fund set aside for follow-on investments."
    )
    
    # --- Main Panel for Bucket Configuration ---
    st.header("Investment Bucket Configuration")
    st.button("Add New Bucket", on_click=add_bucket, use_container_width=True)

    # --- Capital Pool Calculations for UI display ---
    fund_size = model.get('fund_size', 0)
    follow_on_reserve_pct = model.get('follow_on_reserve', 0)
    management_fee_reserve = fund_size * 0.17 
    investable_capital = fund_size - management_fee_reserve
    initial_capital_pool = investable_capital * (1 - follow_on_reserve_pct / 100)
    total_follow_on_pool = investable_capital * (follow_on_reserve_pct / 100)

    # Sort keys to ensure consistent order
    sorted_bucket_keys = sorted(model.get('buckets', {}).keys(), key=int)

    for i, i_str in enumerate(sorted_bucket_keys):
        bucket = model['buckets'][i_str]
        with st.expander(f"Bucket: {bucket.get('name', '')} ({int(bucket.get('percentage', 0))}%)", expanded=True):
            
            c1, c2 = st.columns([3, 1])
            with c1:
                st.text_input("Bucket Name", value=bucket.get('name'), key=f'fm_b_{i_str}_name', label_visibility="collapsed", on_change=update_model_value, args=(['buckets', i_str, 'name'], f'fm_b_{i_str}_name'))
            with c2:
                st.button("Delete", key=f'remove_b_{i_str}', on_click=remove_bucket, args=(i_str,), use_container_width=True)

            alloc_c1, alloc_c2 = st.columns(2)
            with alloc_c1:
                percentage_initial = bucket.get('percentage', 0)
                absolute_initial = initial_capital_pool * (percentage_initial / 100)
                st.slider("Percentage of Initial Capital (%)", 0, 100, value=int(bucket.get('percentage', 0)), key=f'fm_b_{i_str}_perc', on_change=update_model_value, args=(['buckets', i_str, 'percentage'], f'fm_b_{i_str}_perc'))
                st.caption(f"Allocated: **${absolute_initial:.2f}M**")
            with alloc_c2:
                percentage_follow_on = bucket.get('follow_on_allocation_pct', 0)
                absolute_follow_on = total_follow_on_pool * (percentage_follow_on / 100)
                st.slider("Percentage of Follow-on Capital (%)", 0, 100, value=int(bucket.get('follow_on_allocation_pct', 0)), key=f'fm_b_{i_str}_follow_perc', on_change=update_model_value, args=(['buckets', i_str, 'follow_on_allocation_pct'], f'fm_b_{i_str}_follow_perc'))
                st.caption(f"Allocated: **${absolute_follow_on:.2f}M**")


            st.markdown("---")
            st.subheader("Deployment Schedule")
            c1, c2, c3, c4 = st.columns(4)
            c1.number_input("Year 1 (%)", 0, 100, value=bucket.get('deploy_y1'), key=f'fm_b_{i_str}_d1', on_change=update_model_value, args=(['buckets', i_str, 'deploy_y1'], f'fm_b_{i_str}_d1'))
            c2.number_input("Year 2 (%)", 0, 100, value=bucket.get('deploy_y2'), key=f'fm_b_{i_str}_d2', on_change=update_model_value, args=(['buckets', i_str, 'deploy_y2'], f'fm_b_{i_str}_d2'))
            c3.number_input("Year 3 (%)", 0, 100, value=bucket.get('deploy_y3'), key=f'fm_b_{i_str}_d3', on_change=update_model_value, args=(['buckets', i_str, 'deploy_y3'], f'fm_b_{i_str}_d3'))
            c4.number_input("Year 4 (%)", 0, 100, value=bucket.get('deploy_y4'), key=f'fm_b_{i_str}_d4', on_change=update_model_value, args=(['buckets', i_str, 'deploy_y4'], f'fm_b_{i_str}_d4'))

            st.markdown("---")
            st.subheader("Investment Thesis")
            t_c1, t_c2, t_c3 = st.columns(3)
            
            t_c1.number_input("Average Ticket Size ($M)", min_value=0.1, step=0.1, format="%.1f",
                value=float(bucket.get('avg_ticket')), key=f'fm_b_{i_str}_ticket', on_change=update_model_value, args=(['buckets', i_str, 'avg_ticket'], f'fm_b_{i_str}_ticket'))

            t_c2.number_input("Min Entry Valuation ($M)", min_value=0.0, step=1.0, format="%.1f",
                value=float(bucket.get('entry_valuation_min', 0.0)), key=f'fm_b_{i_str}_entry_val_min', on_change=update_model_value, args=(['buckets', i_str, 'entry_valuation_min'], f'fm_b_{i_str}_entry_val_min'))
            
            t_c3.number_input("Max Entry Valuation ($M)", min_value=0.0, step=1.0, format="%.1f",
                value=float(bucket.get('entry_valuation_max', 0.0)), key=f'fm_b_{i_str}_entry_val_max', on_change=update_model_value, args=(['buckets', i_str, 'entry_valuation_max'], f'fm_b_{i_str}_entry_val_max'))
            
            # Calculated ownership
            avg_ticket = bucket.get('avg_ticket', 0)
            min_entry_val = bucket.get('entry_valuation_min', 0.0)
            max_entry_val = bucket.get('entry_valuation_max', 0.0)
            
            min_ownership = (avg_ticket / max_entry_val * 100) if max_entry_val > 0 else 0
            max_ownership = (avg_ticket / min_entry_val * 100) if min_entry_val > 0 else 0
            
            st.info(f"**Calculated Ownership Range:** {min_ownership:.1f}% - {max_ownership:.1f}%")

            # Expected number of initial investments
            absolute_initial_for_bucket = initial_capital_pool * (bucket.get('percentage', 0) / 100)
            expected_initial_investments = (absolute_initial_for_bucket / avg_ticket) if avg_ticket > 0 else 0
            st.info(f"Based on a \\${absolute_initial_for_bucket:.2f}M allocation and \\${avg_ticket:.2f}M average ticket, you can make roughly {expected_initial_investments:.1f} initial investments.")

            st.markdown("---")
            st.subheader("Follow-on Strategy")

            fo_c1, fo_c2, fo_c3, fo_c4 = st.columns(4)
            fo_c1.number_input("Follow-on Prob. (%)", 0, 100,
                value=bucket.get('follow_on_probability', 50),
                key=f'fm_b_{i_str}_foprob', help="Probability of a follow-on round for any investment in this bucket.",
                on_change=update_model_value, args=(['buckets', i_str, 'follow_on_probability'], f'fm_b_{i_str}_foprob')
            )
            fo_c2.number_input("Timing (Yrs after initial)", min_value=0.0, step=0.5, format="%.1f",
                value=float(bucket.get('follow_on_timing', 2.0)),
                key=f'fm_b_{i_str}_fotime',
                on_change=update_model_value, args=(['buckets', i_str, 'follow_on_timing'], f'fm_b_{i_str}_fotime')
            )
            fo_c3.number_input("Size (% of Initial)", min_value=0, step=10,
                value=bucket.get('follow_on_size_pct_of_initial', 200),
                key=f'fm_b_{i_str}_fosize',
                on_change=update_model_value, args=(['buckets', i_str, 'follow_on_size_pct_of_initial'], f'fm_b_{i_str}_fosize')
            )
            fo_c4.number_input("Valuation (x Entry)", min_value=1.0, step=0.1, format="%.1f",
                value=float(bucket.get('follow_on_valuation_multiple', 2.0)),
                key=f'fm_b_{i_str}_foval',
                on_change=update_model_value, args=(['buckets', i_str, 'follow_on_valuation_multiple'], f'fm_b_{i_str}_foval')
            )

            # --- Dynamic Follow-on Calculation ---
            avg_ticket = bucket.get('avg_ticket', 0)
            follow_on_prob_pct = bucket.get('follow_on_probability', 50)
            follow_on_size_pct = bucket.get('follow_on_size_pct_of_initial', 200)
            percentage_follow_on = bucket.get('follow_on_allocation_pct', 0)
            
            absolute_initial_for_bucket = initial_capital_pool * (bucket.get('percentage', 0) / 100)
            expected_initial_investments = (absolute_initial_for_bucket / avg_ticket) if avg_ticket > 0 else 0

            expected_follow_on_investments = expected_initial_investments * (follow_on_prob_pct / 100)
            avg_follow_on_ticket = avg_ticket * (follow_on_size_pct / 100)
            needed_follow_on_capital = expected_follow_on_investments * avg_follow_on_ticket
            
            allocated_follow_on = total_follow_on_pool * (percentage_follow_on / 100)

            message = (
                f"With {expected_initial_investments:.1f} initial investments and a {follow_on_prob_pct:.0f}% follow-on rate, "
                f"you can expect ~{expected_follow_on_investments:.1f} follow-on deals. "
                f"This would require \\${needed_follow_on_capital:.2f}M. You have allocated \\${allocated_follow_on:.2f}M."
            )

            if needed_follow_on_capital > allocated_follow_on and allocated_follow_on > 0:
                st.warning(message)
            else:
                st.info(message)


            st.markdown("---")
            st.subheader("Exit Scenarios")

            # --- Configurable Exit Scenarios ---
            scenarios = bucket.get('scenarios', [])
            for s_idx, scenario in enumerate(scenarios):
                with st.container():
                    # --- ROW 1: Name & Probability ---
                    r1c1, r1c2 = st.columns([3, 7])
                    with r1c1:
                        st.text_input("Scenario Name", value=scenario.get('name', ''), key=f'fm_b_{i_str}_s{s_idx}_name',
                                      label_visibility="collapsed", on_change=update_model_value, args=(['buckets', i_str, 'scenarios', s_idx, 'name'], f'fm_b_{i_str}_s{s_idx}_name'))
                    with r1c2:
                        st.number_input("Probability (%)", 0.0, 100.0, value=float(scenario.get('probability', 0.0)), step=0.1, format="%.1f", key=f'fm_b_{i_str}_s{s_idx}_prob', on_change=update_model_value, args=(['buckets', i_str, 'scenarios', s_idx, 'probability'], f'fm_b_{i_str}_s{s_idx}_prob'))

                    # --- ROW 2: Exit Details ---
                    r2c1, r2c2, r2c3, r2c4 = st.columns([3, 2, 4, 1])
                    with r2c1:
                        st.markdown("**Exit Valuation ($M)**")
                        ev_c1, ev_c2 = st.columns(2)
                        ev_c1.number_input("Min", min_value=0.0, step=0.1, format="%.1f",
                            value=float(scenario.get('exit_valuation_min', 1.0)),
                            key=f'fm_b_{i_str}_s{s_idx}_ev_min', label_visibility="collapsed", on_change=update_model_value, args=(['buckets', i_str, 'scenarios', s_idx, 'exit_valuation_min'], f'fm_b_{i_str}_s{s_idx}_ev_min')
                        )
                        ev_c2.number_input("Max", min_value=0.0, step=0.1, format="%.1f",
                            value=float(scenario.get('exit_valuation_max', 2.0)),
                            key=f'fm_b_{i_str}_s{s_idx}_ev_max', label_visibility="collapsed", on_change=update_model_value, args=(['buckets', i_str, 'scenarios', s_idx, 'exit_valuation_max'], f'fm_b_{i_str}_s{s_idx}_ev_max')
                        )
                    with r2c2:
                        st.markdown("**Exit Dilution (%)**")
                        st.number_input("Dilution (%)", 0, 100,
                            value=scenario.get('exit_dilution_pct', 20),
                            key=f'fm_b_{i_str}_s{s_idx}_dilution', label_visibility="collapsed", on_change=update_model_value, args=(['buckets', i_str, 'scenarios', s_idx, 'exit_dilution_pct'], f'fm_b_{i_str}_s{s_idx}_dilution')
                        )
                    with r2c3:
                        st.slider("Time to Exit (Years)", 1, 15,
                            value=(scenario.get('exit_year_min', 5), scenario.get('exit_year_max', 8)), key=f'fm_b_{i_str}_s{s_idx}_exit', on_change=update_model_value, args=(['buckets', i_str, 'scenarios', s_idx, ['exit_year_min', 'exit_year_max']], f'fm_b_{i_str}_s{s_idx}_exit'))
                    with r2c4:
                        st.button("🗑️", key=f'remove_s_{i_str}_{s_idx}', on_click=remove_scenario, args=(i_str, s_idx), use_container_width=True, help="Remove this scenario")

                if s_idx < len(scenarios) - 1:
                    st.markdown("---")

            st.button("Add Scenario", key=f'add_s_{i_str}', on_click=add_scenario, args=(i_str,), use_container_width=True)
    
        # Add a divider and spacing between bucket cards for better visual separation
        if i < len(sorted_bucket_keys) - 1:
            st.markdown("<br>", unsafe_allow_html=True)
            st.divider()
            st.markdown("<br>", unsafe_allow_html=True)

    
    # Call validation function *after* all UI widgets have been rendered and updated state
    warnings = validate_model_and_get_warnings(model)

    # --- Display all warnings together ---
    if warnings:
        with st.expander("⚠️ Model Configuration Warnings", expanded=True):
            for warning in warnings:
                st.warning(warning)

    st.header("📈 Run Simulation")

    run_disabled = bool(warnings)
    if run_disabled:
        st.info("Please resolve the configuration warnings before running the simulation.")

    if st.button("Run Monte Carlo Simulation", type="primary", disabled=run_disabled):
        with st.spinner("Running main simulation (10,000 iterations)... This might take a moment."):
            main_results_df = run_monte_carlo_simulation(st.session_state.fund_model, 10000)
            st.session_state.simulation_results = main_results_df

        # --- Fund Size Sensitivity Analysis ---
        analysis_results = []
        base_fund_size = st.session_state.fund_model['fund_size']
        
        min_size = base_fund_size - 20
        max_size = base_fund_size + 50
        step = 5
        # Ensure min size is positive
        fund_sizes_to_test = list(range(max(step, min_size), max_size + 1, step))
        if base_fund_size not in fund_sizes_to_test:
            fund_sizes_to_test.append(base_fund_size)
            fund_sizes_to_test.sort()

        status_text = st.empty()
        progress_bar = st.progress(0)
        num_sizes = len(fund_sizes_to_test)

        for i, size in enumerate(fund_sizes_to_test):
            status_text.text(f"Running sensitivity analysis for fund size: ${size}M...")
            
            if size == base_fund_size:
                # Use the high-precision results for the base case
                results_df = main_results_df
            else:
                # Run lower-precision simulation for other sizes
                model_copy = deepcopy(st.session_state.fund_model)
                model_copy['fund_size'] = size
                results_df = run_monte_carlo_simulation(model_copy, 2000)

            # Calculate and store metrics for this fund size
            mean_tvpi = results_df['tvpi'].mean()
            mean_moic = results_df['moic'].mean()
            prob_3x = (results_df['tvpi'] >= 3).mean() * 100
            prob_5x = (results_df['tvpi'] >= 5).mean() * 100
            
            analysis_results.append({
                'fund_size': size,
                'mean_tvpi': mean_tvpi,
                'mean_moic': mean_moic,
                'prob_tvpi_gt_3x': prob_3x,
                'prob_tvpi_gt_5x': prob_5x
            })
            
            progress_bar.progress((i + 1) / num_sizes)
        
        status_text.text("Analysis complete!")
        st.session_state.fund_size_analysis_results = pd.DataFrame(analysis_results)
        progress_bar.empty()
        status_text.empty()
    
    if 'simulation_results' in st.session_state:
        display_simulation_results(st.session_state.simulation_results)


def run_monte_carlo_simulation(fund_model, num_simulations=10000):
    """
    Runs the Monte Carlo simulation for the VC fund model, including cash flow analysis for IRR.
    """
    FUND_LIFE_YEARS = 20  # Fund life for cash flow analysis

    fund_size = fund_model['fund_size']
    follow_on_reserve_pct = fund_model['follow_on_reserve']

    # Reserve capital for management fees from the total fund size.
    # The fee cap (17%) is used to determine the total fee reserve.
    # This matches the cap used in the Net IRR calculation later.
    management_fee_reserve = fund_size * 0.17 
    investable_capital = fund_size - management_fee_reserve

    # Investment pools are now derived from the remaining 'investable_capital'.
    initial_capital_pool = investable_capital * (1 - follow_on_reserve_pct / 100)
    total_follow_on_pool = investable_capital * (follow_on_reserve_pct / 100)

    # Create bucket-specific follow-on pools
    follow_on_sub_pools = {
        i_str: total_follow_on_pool * (bucket.get('follow_on_allocation_pct', 0) / 100)
        for i_str, bucket in fund_model['buckets'].items()
    }
    
    all_simulation_runs = []
    all_portfolios = []


    for sim_idx in range(num_simulations):
        cash_flows = np.zeros(FUND_LIFE_YEARS)
        total_invested_cash = 0
        total_realized_value = 0
        realized_value_by_bucket = {i_str: 0 for i_str in fund_model['buckets']}
        
        # --- NEW: Store detailed portfolio info for this run ---
        portfolio_details = []

        # Track spent capital from each sub-pool
        follow_on_capital_spent_by_bucket = {i_str: 0 for i_str in fund_model['buckets']}
        initial_capital_invested_by_bucket = {i_str: 0 for i_str in fund_model['buckets']}
        initial_investment_count_by_bucket = {i_str: 0 for i_str in fund_model['buckets']}
        follow_on_investment_count_by_bucket = {i_str: 0 for i_str in fund_model['buckets']}

        # Create a list of all initial investments with their deployment years
        all_investments = []
        for i_str, bucket in fund_model['buckets'].items():
            bucket_capital = initial_capital_pool * (bucket['percentage'] / 100)
            avg_ticket = bucket.get('avg_ticket', 0)
            if avg_ticket <= 0: continue
            
            num_investments = int(bucket_capital / avg_ticket)
            
            deploy_pcts = np.array([
                bucket.get('deploy_y1', 0), bucket.get('deploy_y2', 0),
                bucket.get('deploy_y3', 0), bucket.get('deploy_y4', 0)
            ])
            if deploy_pcts.sum() == 0: continue
            deploy_probs = deploy_pcts / deploy_pcts.sum()

            investment_years = np.random.choice([0, 1, 2, 3], size=num_investments, p=deploy_probs)
            for year in investment_years:
                all_investments.append({'bucket_key': i_str, 'bucket': bucket, 'year': year})

        # Process each investment through its lifecycle
        for investment_idx, investment in enumerate(all_investments):
            bucket_key = investment['bucket_key']
            bucket = investment['bucket']
            investment_year = investment['year']
            avg_ticket = bucket.get('avg_ticket', 0)

            # Sample entry valuation for this specific investment
            entry_valuation_min = bucket.get('entry_valuation_min', 0)
            entry_valuation_max = bucket.get('entry_valuation_max', 0)
            entry_valuation = np.random.uniform(entry_valuation_min, entry_valuation_max) if entry_valuation_max > entry_valuation_min else entry_valuation_min

            # Track initial investment stats
            initial_capital_invested_by_bucket[bucket_key] += avg_ticket
            initial_investment_count_by_bucket[bucket_key] += 1

            # Initial investment cash flow
            cash_flows[investment_year] -= avg_ticket
            total_invested_cash += avg_ticket

            # --- Ownership and Return Calculation ---
            initial_ownership_pct = (avg_ticket / entry_valuation * 100) if entry_valuation > 0 else 0
            
            # Handle follow-on investment based on bucket-level strategy
            follow_on_investment = 0
            follow_on_ownership_pct = 0
            did_follow_on = False
            follow_on_prob = bucket.get('follow_on_probability', 0)

            if np.random.uniform(0, 100) < follow_on_prob:
                # Check against the specific bucket's follow-on pool
                follow_on_pool_for_bucket = follow_on_sub_pools.get(bucket_key, 0)
                spent_from_pool = follow_on_capital_spent_by_bucket.get(bucket_key, 0)
                
                follow_on_size_pct = bucket.get('follow_on_size_pct_of_initial', 0)
                follow_on_amount = avg_ticket * (follow_on_size_pct / 100)

                if spent_from_pool + follow_on_amount <= follow_on_pool_for_bucket:
                    follow_on_timing = bucket.get('follow_on_timing', 2.0)
                    follow_on_year = investment_year + follow_on_timing
                    
                    # Ensure year is an integer for indexing
                    if int(follow_on_year) < FUND_LIFE_YEARS:
                        did_follow_on = True
                        follow_on_investment = follow_on_amount
                        follow_on_capital_spent_by_bucket[bucket_key] += follow_on_amount
                        follow_on_investment_count_by_bucket[bucket_key] += 1
                        cash_flows[int(follow_on_year)] -= follow_on_investment
                        total_invested_cash += follow_on_investment
                        
                        # Calculate ownership from follow-on
                        follow_on_val_multiple = bucket.get('follow_on_valuation_multiple', 1.0)
                        follow_on_valuation = entry_valuation * follow_on_val_multiple
                        if follow_on_valuation > 0:
                            follow_on_ownership_pct = (follow_on_investment / follow_on_valuation * 100)

            # Determine outcome
            scenarios = bucket.get('scenarios', [])
            if not scenarios: continue

            probs = np.array([s.get('probability', 0) for s in scenarios], dtype=float)
            if probs.sum() == 0: continue
            probs /= probs.sum()

            chosen_scenario_index = np.random.choice(len(scenarios), p=probs)
            chosen_scenario = scenarios[chosen_scenario_index]

            # Determine exit valuation
            exit_valuation = np.random.uniform(
                chosen_scenario.get('exit_valuation_min', 0.0),
                chosen_scenario.get('exit_valuation_max', 0.0)
            )
            
            total_ownership_pct_before_dilution = initial_ownership_pct + follow_on_ownership_pct
            
            # Handle exit and realized value
            time_to_exit = np.random.randint(
                chosen_scenario.get('exit_year_min', 5), 
                chosen_scenario.get('exit_year_max', 8) + 1
            )
            exit_year = investment_year + time_to_exit

            realized_value = 0
            final_ownership_pct = 0
            status = "Active" # Default status

            if exit_year < FUND_LIFE_YEARS:
                # Apply exit dilution
                exit_dilution_pct = chosen_scenario.get('exit_dilution_pct', 20)
                final_ownership_pct = total_ownership_pct_before_dilution * (1 - exit_dilution_pct / 100)
                
                realized_value = (final_ownership_pct / 100) * exit_valuation
                realized_value_by_bucket[bucket_key] += realized_value
                cash_flows[exit_year] += realized_value
                total_realized_value += realized_value
                
                status = "Exited" if exit_valuation > 0 else "Failed"

            # --- NEW: Store detailed company data ---
            portfolio_details.append({
                'company_id': f"Company {investment_idx + 1}",
                'investment_year': investment_year + 1, # Use 1-based indexing for display
                'stage': bucket.get('name', 'N/A'),
                'initial_check': avg_ticket,
                'initial_ownership': initial_ownership_pct,
                'entry_valuation': entry_valuation,
                'follow_on': "Yes" if did_follow_on else "No",
                'follow_on_check': follow_on_investment,
                'ownership_after_follow_on': total_ownership_pct_before_dilution,
                'final_ownership_at_exit': final_ownership_pct,
                'status': status,
                'exit_year': exit_year + 1 if status != "Active" else None,
                'exit_valuation': exit_valuation if status != "Active" else None,
                'net_return': realized_value,
                'exit_scenario': chosen_scenario.get('name', 'N/A')
            })

        # Calculate metrics for the simulation run
        moic = total_realized_value / total_invested_cash if total_invested_cash > 0 else 0
        tvpi = total_realized_value / fund_size if fund_size > 0 else 0
        
        # --- IRR Calculations (Gross and Net) ---
        gross_irr = np.nan
        net_irr = np.nan
        
        try:
            # 1. Calculate Gross IRR (based on fund's direct cash flows)
            gross_irr = npf.irr(cash_flows)

            # 2. Calculate Net IRR (from LP's perspective with fees and carry)
            lp_net_cash_flows = np.zeros(FUND_LIFE_YEARS)
            total_contributions = 0
            lp_capital_returned = 0
            annual_fee = fund_size * 0.02 # 2% management fee
            total_fees_paid = 0.0
            max_total_fees = fund_size * 0.17 # Cap at 17%

            for year in range(FUND_LIFE_YEARS):
                # Outflows for LP: Investments + Fees
                investment_in_year = cash_flows[year] if cash_flows[year] < 0 else 0
                
                fee_in_year = 0.0
                if year < 10 and total_fees_paid < max_total_fees:
                    fee_to_charge = min(annual_fee, max_total_fees - total_fees_paid)
                    fee_in_year = -fee_to_charge
                    total_fees_paid += fee_to_charge
                
                lp_outflow = investment_in_year + fee_in_year
                lp_net_cash_flows[year] += lp_outflow
                total_contributions += -lp_outflow

                # Inflows for LP: Distributions from exits with waterfall logic
                distribution_in_year = cash_flows[year] if cash_flows[year] > 0 else 0
                if distribution_in_year > 0:
                    # First, return all contributed capital to LPs
                    capital_to_return_hurdle = total_contributions - lp_capital_returned
                    dist_for_capital_return = min(distribution_in_year, capital_to_return_hurdle)
                    
                    lp_net_cash_flows[year] += dist_for_capital_return
                    lp_capital_returned += dist_for_capital_return
                    
                    # Then, split remaining profit 80/20
                    profit_distribution = distribution_in_year - dist_for_capital_return
                    if profit_distribution > 0:
                        lp_share_of_profit = profit_distribution * 0.80 # 80% to LPs
                        lp_net_cash_flows[year] += lp_share_of_profit

            net_irr = npf.irr(lp_net_cash_flows)

        except ValueError:
            # If IRR calculation fails for either, set both to a failure value
            gross_irr = -1.0
            net_irr = -1.0

        run_data = {
            'moic': moic, 'tvpi': tvpi, 
            'gross_irr': gross_irr, 'net_irr': net_irr,
            'total_invested': total_invested_cash, 'total_realized': total_realized_value
        }
        for i_str in fund_model['buckets'].keys():
            run_data[f'initial_invested_b{i_str}'] = initial_capital_invested_by_bucket[i_str]
            run_data[f'initial_count_b{i_str}'] = initial_investment_count_by_bucket[i_str]
            run_data[f'follow_on_invested_b{i_str}'] = follow_on_capital_spent_by_bucket[i_str]
            run_data[f'follow_on_count_b{i_str}'] = follow_on_investment_count_by_bucket[i_str]
            run_data[f'realized_b{i_str}'] = realized_value_by_bucket.get(i_str, 0)
        
        all_simulation_runs.append(run_data)
        all_portfolios.append(portfolio_details)

    # --- NEW: Return both simulation results and portfolio details ---
    results_df = pd.DataFrame(all_simulation_runs)
    results_df['portfolio_details'] = all_portfolios
    
    return results_df


def display_simulation_results(results_df):
    """
    Displays the results of the Monte Carlo simulation.
    """
    st.header("📈 Simulation Results")

    # --- Metrics ---
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    mean_tvpi = results_df['tvpi'].mean()
    median_tvpi = results_df['tvpi'].median()
    mean_moic = results_df['moic'].mean()
    prob_3x = (results_df['tvpi'] >= 3).mean() * 100
    prob_5x = (results_df['tvpi'] >= 5).mean() * 100
    prob_loss = (results_df['tvpi'] < 1).mean() * 100

    col1.metric("Mean TVPI", f"{mean_tvpi:.2f}x")
    col2.metric("Mean MOIC", f"{mean_moic:.2f}x")
    col3.metric("Median TVPI", f"{median_tvpi:.2f}x")
    col4.metric("P(Loss of Capital)", f"{prob_loss:.1f}%")
    col5.metric("P(TVPI > 3x)", f"{prob_3x:.1f}%")
    col6.metric("P(TVPI > 5x)", f"{prob_5x:.1f}%")

    # --- IRR Metrics ---
    # Filter out failed IRR calculations for metrics
    valid_gross_irr = results_df['gross_irr'].dropna()
    valid_net_irr = results_df['net_irr'].dropna()

    mean_gross_irr = valid_gross_irr.mean() * 100
    median_gross_irr = valid_gross_irr.median() * 100
    prob_gross_irr_25 = (valid_gross_irr >= 0.25).mean() * 100
    
    mean_net_irr = valid_net_irr.mean() * 100
    median_net_irr = valid_net_irr.median() * 100
    prob_net_irr_25 = (valid_net_irr >= 0.25).mean() * 100

    st.subheader("Fund IRR (Internal Rate of Return)")
    st.info("Net IRR is calculated assuming a '2 and 20' fund structure (2% annual management fee for 10 years, capped at 17% of total fund size, and 20% carried interest).", icon="ℹ️")

    irr_c1, irr_c2 = st.columns(2)
    with irr_c1:
        st.markdown("##### Gross IRR (Fund-level)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Gross IRR", f"{mean_gross_irr:.1f}%")
        col2.metric("Median Gross IRR", f"{median_gross_irr:.1f}%")
        col3.metric("P(Gross > 25%)", f"{prob_gross_irr_25:.1f}%")

    with irr_c2:
        st.markdown("##### Net IRR (LP-level)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Net IRR", f"{mean_net_irr:.1f}%")
        col2.metric("Median Net IRR", f"{median_net_irr:.1f}%")
        col3.metric("P(Net > 25%)", f"{prob_net_irr_25:.1f}%")

    # --- Capital Deployment Analysis ---
    st.subheader("Capital Deployment Analysis")
    with st.expander("Show Detailed Deployment Statistics by Bucket"):
        model = st.session_state.fund_model
        
        # Recalculate capital pools to get allocated amounts
        fund_size = model['fund_size']
        follow_on_reserve_pct = model['follow_on_reserve']
        management_fee_reserve = fund_size * 0.17 
        investable_capital = fund_size - management_fee_reserve
        initial_capital_pool = investable_capital * (1 - follow_on_reserve_pct / 100)
        total_follow_on_pool = investable_capital * (follow_on_reserve_pct / 100)
        follow_on_sub_pools = {
            i_str: total_follow_on_pool * (bucket.get('follow_on_allocation_pct', 0) / 100)
            for i_str, bucket in model['buckets'].items()
        }

        # --- Full Fund Stats ---
        st.markdown("#### Full Fund Deployment")
        mean_total_invested = results_df['total_invested'].mean()
        st.metric(
            label="Total Capital Deployed (vs. Investable)",
            value=f"${mean_total_invested:.2f}M / ${investable_capital:.2f}M",
            help="Investable capital is the fund size minus a 17% reserve for management fees."
        )

        sorted_bucket_keys = sorted(model.get('buckets', {}).keys(), key=int)
        total_initial_investments = sum(results_df[f'initial_count_b{i_str}'].mean() for i_str in sorted_bucket_keys)
        total_follow_on_investments = sum(results_df[f'follow_on_count_b{i_str}'].mean() for i_str in sorted_bucket_keys)

        col1, col2 = st.columns(2)
        col1.metric("Total Initial Investments", f"{total_initial_investments:.1f}")
        col2.metric("Total Follow-on Investments", f"{total_follow_on_investments:.1f}")

        st.markdown("---")
        st.markdown("#### Deployment by Bucket")

        # Display stats for each bucket
        for i_str in sorted_bucket_keys:
            bucket = model['buckets'][i_str]
            st.markdown(f"##### Bucket: {bucket.get('name', '')}")

            # Calculate allocated amounts for this bucket
            allocated_initial = initial_capital_pool * (bucket.get('percentage', 0) / 100)
            allocated_follow_on = follow_on_sub_pools.get(i_str, 0)
            
            # Get mean results from the simulation dataframe
            mean_invested_initial = results_df[f'initial_invested_b{i_str}'].mean()
            mean_count_initial = results_df[f'initial_count_b{i_str}'].mean()
            mean_invested_follow_on = results_df[f'follow_on_invested_b{i_str}'].mean()
            mean_count_follow_on = results_df[f'follow_on_count_b{i_str}'].mean()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Initial Investments**")
                st.metric(
                    label="Capital Deployed (vs. Allocated)",
                    value=f"${mean_invested_initial:.2f}M / ${allocated_initial:.2f}M"
                )
                st.metric(
                    label="Number of Investments",
                    value=f"{mean_count_initial:.1f}",
                    help=f"The average number of initial investments made from this bucket in the simulation."
                )

            with col2:
                st.markdown("**Follow-on Investments**")
                st.metric(
                    label="Capital Deployed (vs. Allocated)",
                    value=f"${mean_invested_follow_on:.2f}M / ${allocated_follow_on:.2f}M"
                )
                st.metric(
                    label="Number of Investments",
                    value=f"{mean_count_follow_on:.1f}",
                    help=f"The average number of follow-on investments made from this bucket's reserve."
                )
            
            if i_str != sorted_bucket_keys[-1]:
                 st.markdown("---")

    # --- Charts ---
    # Define tabs
    tab_titles = ["TVPI Distribution", "MOIC Distribution", "Gross IRR Distribution", "Net IRR Distribution"]
    if 'fund_size_analysis_results' in st.session_state:
        tab_titles.append("Fund Size Analysis")
    
    # --- NEW: Add Example Portfolio Tab if data exists ---
    if 'portfolio_details' in results_df.columns:
        tab_titles.append("Example Portfolios")

    tabs = st.tabs(tab_titles)

    with tabs[0]: # TVPI Distribution
        model = st.session_state.fund_model
        sorted_bucket_keys = sorted(model.get('buckets', {}).keys(), key=int)

        # 1. Calculate all TVPI series to determine shared axis ranges
        main_tvpi = results_df['tvpi']
        bucket_tvpis = {}
        for i_str in sorted_bucket_keys:
            bucket = model['buckets'][i_str]
            invested_col_initial = f'initial_invested_b{i_str}'
            invested_col_follow_on = f'follow_on_invested_b{i_str}'
            realized_col = f'realized_b{i_str}'
            
            if realized_col in results_df.columns:
                total_invested_in_bucket = results_df[invested_col_initial] + results_df[invested_col_follow_on]
                bucket_tvpi = (results_df[realized_col] / total_invested_in_bucket).replace([np.inf, -np.inf], 0).fillna(0)
                bucket_tvpis[i_str] = bucket_tvpi

        # 2. Determine shared Y-axis range by pre-calculating histogram heights
        all_tvpi_series = [main_tvpi] + list(bucket_tvpis.values())
        max_y = 0
        # Use a high but fixed TVPI value for consistent binning to find max Y
        hist_range_max = 50 
        num_bins = 200 # Consistent number of bins for height calculation
        
        for series in all_tvpi_series:
            clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
            if not clean_series.empty:
                counts, _ = np.histogram(clean_series, bins=num_bins, range=(0, hist_range_max))
                # Normalize to percentage of total
                percentages = (counts / len(clean_series)) * 100 if len(clean_series) > 0 else counts
                if len(percentages) > 0:
                    max_y = max(max_y, percentages.max())
        
        # Add padding to the top of the Y-axis
        y_axis_range = [0, max_y * 1.15] if max_y > 0 else [0, 1]
        # Set a fixed, scrollable X-axis range as requested
        x_axis_range = [0, 15]
        
        # Define shared histogram binning for consistency
        histogram_bins = dict(
            start=0,
            end=hist_range_max, # Using the same 50 as for y-axis calculation
            size=hist_range_max / num_bins # e.g., 50 / 200 = 0.25
        )
        
        # 4. Render main chart with shared ranges
        st.subheader("Distribution of Fund Returns (TVPI)")
        fig_tvpi = go.Figure()
        fig_tvpi.add_trace(go.Histogram(x=main_tvpi, xbins=histogram_bins, name='Distribution', histnorm='percent'))
        fig_tvpi.add_vline(x=mean_tvpi, line_width=2, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {mean_tvpi:.2f}x", annotation_position="top right")
        fig_tvpi.update_layout(
            title="Distribution of Fund Return Multiples (Total Value / Fund Size)",
            xaxis_title="Fund Return Multiple (TVPI)", yaxis_title="Probability (%)", bargap=0.1,
            xaxis_range=x_axis_range,
            yaxis_range=y_axis_range
        )
        st.plotly_chart(fig_tvpi, use_container_width=True)

        st.markdown("---")
        st.subheader("TVPI Distribution by Investment Bucket")
        
        # 5. Render bucket charts with shared ranges
        for i_str in sorted_bucket_keys:
            if i_str not in bucket_tvpis:
                continue

            bucket = model['buckets'][i_str]
            bucket_name = bucket.get('name', f'Bucket {i_str}')
            bucket_tvpi = bucket_tvpis[i_str]
            mean_bucket_tvpi = bucket_tvpi.mean()

            fig_bucket_tvpi = go.Figure()
            fig_bucket_tvpi.add_trace(
                go.Histogram(
                    x=bucket_tvpi, 
                    xbins=histogram_bins, 
                    name='Distribution', 
                    histnorm='percent',
                    marker_color='purple'
                )
            )
            fig_bucket_tvpi.add_vline(
                x=mean_bucket_tvpi, 
                line_width=2, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Mean: {mean_bucket_tvpi:.2f}x", 
                annotation_position="top right"
            )
            fig_bucket_tvpi.update_layout(
                title=f"TVPI Distribution for '{bucket_name}'",
                xaxis_title=f"Bucket TVPI Multiple",
                yaxis_title="Probability (%)",
                bargap=0.1,
                xaxis_range=x_axis_range,
                yaxis_range=y_axis_range
            )
            st.plotly_chart(fig_bucket_tvpi, use_container_width=True)

    with tabs[1]: # MOIC Distribution
        st.subheader("Distribution of Investment Returns (MOIC)")
        
        # Determine shared ranges for MOIC charts
        main_moic = results_df['moic']
        bucket_moics = {}
        for i_str in sorted_bucket_keys:
            bucket = model['buckets'][i_str]
            invested_col_initial = f'initial_invested_b{i_str}'
            invested_col_follow_on = f'follow_on_invested_b{i_str}'
            realized_col = f'realized_b{i_str}'
            
            if realized_col in results_df.columns:
                total_invested_in_bucket = results_df[invested_col_initial] + results_df[invested_col_follow_on]
                bucket_moic = (results_df[realized_col] / total_invested_in_bucket).replace([np.inf, -np.inf], 0).fillna(0)
                bucket_moics[i_str] = bucket_moic
        
        all_moic_series = [main_moic] + list(bucket_moics.values())
        max_y_moic = 0
        hist_range_max_moic = 50 
        num_bins_moic = 200 
        
        for series in all_moic_series:
            clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
            if not clean_series.empty:
                counts, _ = np.histogram(clean_series, bins=num_bins_moic, range=(0, hist_range_max_moic))
                percentages = (counts / len(clean_series)) * 100 if len(clean_series) > 0 else counts
                if len(percentages) > 0:
                    max_y_moic = max(max_y_moic, percentages.max())
        
        y_axis_range_moic = [0, max_y_moic * 1.15] if max_y_moic > 0 else [0, 1]
        x_axis_range_moic = [0, 15]

        histogram_bins_moic = dict(start=0, end=hist_range_max_moic, size=hist_range_max_moic / num_bins_moic)

        # Main MOIC chart
        mean_moic = main_moic.mean()
        fig_moic = go.Figure()
        fig_moic.add_trace(go.Histogram(x=main_moic, xbins=histogram_bins_moic, name='Distribution', histnorm='percent', marker_color='green'))
        fig_moic.add_vline(x=mean_moic, line_width=2, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {mean_moic:.2f}x", annotation_position="top right")
        fig_moic.update_layout(
            title="Distribution of Fund Return Multiples (Total Value / Invested Capital)",
            xaxis_title="Fund Return Multiple (MOIC)", yaxis_title="Probability (%)", bargap=0.1,
            xaxis_range=x_axis_range_moic,
            yaxis_range=y_axis_range_moic
        )
        st.plotly_chart(fig_moic, use_container_width=True)

        st.markdown("---")
        st.subheader("MOIC Distribution by Investment Bucket")

        # Bucket MOIC charts
        for i_str in sorted_bucket_keys:
            if i_str not in bucket_moics:
                continue
            bucket_name = model['buckets'][i_str].get('name', f'Bucket {i_str}')
            bucket_moic = bucket_moics[i_str]
            mean_bucket_moic = bucket_moic.mean()

            fig_bucket_moic = go.Figure()
            fig_bucket_moic.add_trace(go.Histogram(x=bucket_moic, xbins=histogram_bins_moic, name='Distribution', histnorm='percent', marker_color='orange'))
            fig_bucket_moic.add_vline(x=mean_bucket_moic, line_width=2, line_dash="dash", line_color="red",
                                      annotation_text=f"Mean: {mean_bucket_moic:.2f}x", annotation_position="top right")
            fig_bucket_moic.update_layout(
                title=f"MOIC Distribution for '{bucket_name}'",
                xaxis_title="Bucket MOIC Multiple", yaxis_title="Probability (%)", bargap=0.1,
                xaxis_range=x_axis_range_moic,
                yaxis_range=y_axis_range_moic
            )
            st.plotly_chart(fig_bucket_moic, use_container_width=True)


    with tabs[2]: # Gross IRR Distribution
        st.subheader("Distribution of Fund Gross IRR")
        fig_irr = go.Figure()
        # Multiply by 100 for percentage representation
        fig_irr.add_trace(go.Histogram(x=valid_gross_irr * 100, nbinsx=50, name='Distribution', histnorm='percent'))
        fig_irr.add_vline(x=mean_gross_irr, line_width=2, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {mean_gross_irr:.1f}%", annotation_position="top right")
        fig_irr.update_layout(
            title="Distribution of Gross Internal Rate of Return (IRR)",
            xaxis_title="Gross IRR (%)", yaxis_title="Probability (%)", bargap=0.1
        )
        st.plotly_chart(fig_irr, use_container_width=True)

    with tabs[3]: # Net IRR Distribution
        st.subheader("Distribution of Fund Net IRR")
        fig_irr_net = go.Figure()
        # Multiply by 100 for percentage representation
        fig_irr_net.add_trace(go.Histogram(x=valid_net_irr * 100, nbinsx=50, name='Distribution', histnorm='percent'))
        fig_irr_net.add_vline(x=mean_net_irr, line_width=2, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {mean_net_irr:.1f}%", annotation_position="top right")
        fig_irr_net.update_layout(
            title="Distribution of Net Internal Rate of Return (IRR) for LPs",
            xaxis_title="Net IRR (%)", yaxis_title="Probability (%)", bargap=0.1
        )
        st.plotly_chart(fig_irr_net, use_container_width=True)
    
    if "Fund Size Analysis" in tab_titles:
        with tabs[4]:
            create_analysis_tab()
    
    # --- NEW: Logic for Example Portfolio Tab ---
    if "Example Portfolios" in tab_titles:
        with tabs[tab_titles.index("Example Portfolios")]:
            create_example_portfolios_tab(results_df, st.session_state.fund_model)


    # with st.expander("View Raw Simulation Data"):
    #     st.dataframe(results_df.style.format({
    #         'moic': '{:.2f}x',
    #         'tvpi': '{:.2f}x',
    #         'gross_irr': '{:.2%}',
    #         'net_irr': '{:.2%}',
    #         'total_invested': '${:,.2f}M',
    #         'total_realized': '${:,.2f}M',
    #     }))


def create_example_portfolios_tab(results_df, fund_model):
    """
    Creates a set of tabs to display different example portfolios (e.g., median, upside cases).
    """
    st.header("✨ Representative Portfolio Outcomes")
    st.info(
        "This section displays several complete portfolios from the simulation. Each portfolio is chosen "
        "to represent a specific outcome scenario (e.g., median, bull case) based on its TVPI ranking.",
        icon="ℹ️"
    )

    percentiles = {
        "Median Case (50th Percentile)": 0.50,
        "Upper-Median Case (65th Percentile)": 0.65,
        "Bull Case (90th Percentile)": 0.90,
    }

    portfolio_tabs = st.tabs(percentiles.keys())

    for i, (tab_title, percentile) in enumerate(percentiles.items()):
        with portfolio_tabs[i]:
            create_single_portfolio_view(results_df, fund_model, percentile, tab_title)


def create_single_portfolio_view(results_df, fund_model, percentile, tab_title):
    """
    Creates the content for a single example portfolio view based on a given TVPI percentile.
    """
    # 1. Find the portfolio closest to the target TVPI percentile
    target_tvpi = results_df['tvpi'].quantile(percentile)
    closest_run = results_df.iloc[(results_df['tvpi'] - target_tvpi).abs().argsort()[0]]
    
    portfolio_data = closest_run['portfolio_details']
    
    if not portfolio_data:
        st.warning(f"The selected portfolio for the {tab_title} has no investments.")
        return

    # Convert list of dicts to DataFrame for display
    portfolio_df = pd.DataFrame(portfolio_data)

    # 2. Display Key Metrics for this specific portfolio
    st.subheader("Portfolio Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Portfolio TVPI", f"{closest_run['tvpi']:.2f}x", help=f"Target TVPI for this scenario ({percentile:.0%}) was {target_tvpi:.2f}x")
    col2.metric("Portfolio MOIC", f"{closest_run['moic']:.2f}x")
    col3.metric("Gross IRR", f"{closest_run['gross_irr']:.1%}")
    col4.metric("Net IRR", f"{closest_run['net_irr']:.1%}")

    # 3. Display Deployment Summary for this portfolio
    st.subheader("Deployment Summary")
    total_deployed = portfolio_df['initial_check'].sum() + portfolio_df['follow_on_check'].sum()
    total_proceeds = portfolio_df['net_return'].sum()
    
    sum_c1, sum_c2 = st.columns(2)
    sum_c1.metric("Total Capital Deployed", f"${total_deployed:.2f}M")
    sum_c2.metric("Total Proceeds", f"${total_proceeds:.2f}M")

    with st.expander("Show Deployment Statistics by Bucket for this Portfolio"):
        # Calculate allocated capital amounts from the model
        fund_size = fund_model['fund_size']
        follow_on_reserve_pct = fund_model['follow_on_reserve']
        management_fee_reserve = fund_size * 0.17
        investable_capital = fund_size - management_fee_reserve
        initial_capital_pool = investable_capital * (1 - follow_on_reserve_pct / 100)
        total_follow_on_pool = investable_capital * (follow_on_reserve_pct / 100)
        
        bucket_allocations = []
        for i_str, bucket in fund_model['buckets'].items():
            allocated_initial = initial_capital_pool * (bucket.get('percentage', 0) / 100)
            allocated_follow_on = total_follow_on_pool * (bucket.get('follow_on_allocation_pct', 0) / 100)
            bucket_allocations.append({
                'stage': bucket.get('name'),
                'allocated_initial': allocated_initial,
                'allocated_follow_on': allocated_follow_on
            })
        allocations_df = pd.DataFrame(bucket_allocations)

        # Calculate actual deployed stats from the portfolio
        bucket_stats_df = portfolio_df.groupby('stage').agg(
            deployed_initial=('initial_check', 'sum'),
            count_initial=('company_id', 'size'),
            deployed_follow_on=('follow_on_check', 'sum'),
            count_follow_on=('follow_on', lambda x: (x == 'Yes').sum())
        ).reset_index()
        
        # Merge allocated with actual
        summary_df = pd.merge(allocations_df, bucket_stats_df, on='stage', how='left').fillna(0)
        
        for _, row in summary_df.iterrows():
            st.markdown(f"##### Bucket: {row['stage']}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Initial Investments**")
                st.metric(
                    "Capital Deployed (vs. Allocated)",
                    f"${row['deployed_initial']:.2f}M / ${row['allocated_initial']:.2f}M"
                )
                st.metric("Number of Investments", f"{int(row['count_initial'])}")
            with c2:
                st.markdown("**Follow-on Investments**")
                st.metric(
                    "Capital Deployed (vs. Allocated)",
                    f"${row['deployed_follow_on']:.2f}M / ${row['allocated_follow_on']:.2f}M"
                )
                st.metric("Number of Investments", f"{int(row['count_follow_on'])}")
            st.markdown("---")

    # 4. Display Exit Distribution Chart
    st.subheader("Exit Distribution")
    exited_df = portfolio_df[portfolio_df['status'].isin(['Exited', 'Failed'])].copy()

    scenario_map = {
        "Failure": "0", "$10M Exit": "10M", "$50M Exit": "50M",
        "$100M Exit": "100M", "$500M Exit": "500M", "$1B+ Exit": "1B",
        "Base Case": "Base", "Home Run": "Home Run" # For strategic buckets
    }
    ordered_scenarios = list(scenario_map.keys())
    
    exited_df['exit_category'] = pd.Categorical(
        exited_df['exit_scenario'],
        categories=ordered_scenarios,
        ordered=True
    )
    exit_counts = exited_df['exit_category'].value_counts().sort_index()
    
    chart_data = pd.DataFrame({
        'scenario': exit_counts.index,
        'count': exit_counts.values
    })
    chart_data['label'] = chart_data['scenario'].map(scenario_map)
    
    if not chart_data.empty:
        fig = go.Figure(data=[
            go.Bar(
                x=chart_data['label'], 
                y=chart_data['count'],
                text=chart_data['count'],
                textposition='inside',
                marker_color='royalblue',
                textfont=dict(color='white', size=14, family='Arial, sans-serif')
            )
        ])
        fig.update_layout(
            title_text='Distribution of Company Outcomes by Exit Scenario',
            xaxis_title="Exit Valuation Estimate",
            yaxis_title="Number of Companies",
            yaxis=dict(range=[0, chart_data['count'].max() * 1.15])
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No companies have exited in this portfolio yet.")

    # 5. Display Portfolio Company Table
    st.subheader("Company Investment Details")
    
    # Reorder columns for logical presentation
    display_columns = [
        'company_id', 'investment_year', 'stage', 'status', 
        'initial_check', 'initial_ownership',
        'follow_on', 'follow_on_check', 'ownership_after_follow_on', 
        'final_ownership_at_exit', 'exit_year', 'exit_valuation', 'net_return'
    ]
    portfolio_df = portfolio_df[display_columns]

    # Format the DataFrame for better readability and hide the index
    styler = portfolio_df.style.format({
            'initial_check': "${:,.2f}M",
            'initial_ownership': "{:.2f}%",
            'entry_valuation': "${:,.1f}M",
            'follow_on_check': "${:,.2f}M",
            'ownership_after_follow_on': "{:.2f}%",
            'final_ownership_at_exit': "{:.2f}%",
            'exit_valuation': "${:,.1f}M",
            'net_return': "${:,.2f}M",
        }).set_properties(**{'text-align': 'left'}) \
          .set_table_styles([dict(selector='th', props=[('text-align', 'left')])]) \
          .hide(axis="index")

    # Calculate dynamic height: 35px per row + 35px for the header
    table_height = (len(portfolio_df) + 1) * 35
    st.dataframe(styler, height=table_height)


def create_analysis_tab():
    st.subheader("Fund Size Sensitivity Analysis")
    st.info("This analysis shows how key return metrics change based on the total fund size. The point corresponding to your configured fund size uses the high-precision 10k simulation, while other points use a faster 2k simulation.", icon="ℹ️")
    
    analysis_df = st.session_state.fund_size_analysis_results
    base_fund_size = st.session_state.fund_model['fund_size']

    # Chart 1: Mean TVPI vs. Fund Size
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=analysis_df['fund_size'], y=analysis_df['mean_tvpi'], mode='lines+markers', name='Mean TVPI'))
    # Highlight the base fund size
    base_point = analysis_df[analysis_df['fund_size'] == base_fund_size]
    if not base_point.empty:
        fig1.add_trace(go.Scatter(x=base_point['fund_size'], y=base_point['mean_tvpi'], mode='markers', marker=dict(color='red', size=10), name='Your Fund'))
    fig1.add_hline(y=1, line_width=2, line_dash="dash", line_color="red")
    fig1.update_layout(
        title="Mean TVPI vs. Fund Size",
        xaxis_title="Fund Size ($M)",
        yaxis_title="Mean TVPI (x)",
        yaxis_range=[0, None]
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: P(TVPI > 3x) vs. Fund Size
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=analysis_df['fund_size'], y=analysis_df['prob_tvpi_gt_3x'], mode='lines+markers', name='P(TVPI > 3x)'))
    if not base_point.empty:
        fig2.add_trace(go.Scatter(x=base_point['fund_size'], y=base_point['prob_tvpi_gt_3x'], mode='markers', marker=dict(color='red', size=10), name='Your Fund'))
    fig2.update_layout(
        title="Probability of >3x Return vs. Fund Size",
        xaxis_title="Fund Size ($M)",
        yaxis_title="Probability (%)",
        yaxis_range=[0, None]
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Chart 3: P(TVPI > 5x) vs. Fund Size
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=analysis_df['fund_size'], y=analysis_df['prob_tvpi_gt_5x'], mode='lines+markers', name='P(TVPI > 5x)'))
    if not base_point.empty:
        fig3.add_trace(go.Scatter(x=base_point['fund_size'], y=base_point['prob_tvpi_gt_5x'], mode='markers', marker=dict(color='red', size=10), name='Your Fund'))
    fig3.update_layout(
        title="Probability of >5x Return vs. Fund Size",
        xaxis_title="Fund Size ($M)",
        yaxis_title="Probability (%)",
        yaxis_range=[0, None]
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Chart 4: Mean MOIC vs. Fund Size
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=analysis_df['fund_size'], y=analysis_df['mean_moic'], mode='lines+markers', name='Mean MOIC'))
    if not base_point.empty:
        fig4.add_trace(go.Scatter(x=base_point['fund_size'], y=base_point['mean_moic'], mode='markers', marker=dict(color='red', size=10), name='Your Fund'))
    fig4.add_hline(y=1, line_width=2, line_dash="dash", line_color="red")
    fig4.update_layout(
        title="Mean MOIC vs. Fund Size",
        xaxis_title="Fund Size ($M)",
        yaxis_title="Mean MOIC (x)",
        yaxis_range=[0, None]
    )
    st.plotly_chart(fig4, use_container_width=True)

def validate_model_and_get_warnings(model):
    """
    Validates the entire fund model configuration and returns a list of warning strings.
    """
    warnings = []

    # 1. Check if total bucket allocation is 100%
    total_percentage = sum(int(b.get('percentage', 0)) for b in model.get('buckets', {}).values())
    if total_percentage != 100:
        warnings.append(f"Total bucket allocation is {total_percentage}%, but should be 100%.")

    # 2. Check follow-on allocation total
    total_follow_on_percentage = sum(int(b.get('follow_on_allocation_pct', 0)) for b in model.get('buckets', {}).values())
    if total_follow_on_percentage != 100:
        warnings.append(f"Total follow-on capital allocation is {total_follow_on_percentage}%, but should be 100%.")

    # 3. Check settings for each bucket
    for i_str, bucket in model.get('buckets', {}).items():
        bucket_name = bucket.get('name', f'Bucket {int(i_str)+1}')
        
        # Check deployment schedule total
        deployment_sum = sum([
            bucket.get('deploy_y1', 0), bucket.get('deploy_y2', 0),
            bucket.get('deploy_y3', 0), bucket.get('deploy_y4', 0)
        ])
        if deployment_sum != 100:
            warnings.append(f"In '{bucket_name}', the deployment schedule sums to {deployment_sum}%, but should be 100%.")
            
        # Check exit scenario probability total
        prob_sum = sum(s.get('probability', 0) for s in bucket.get('scenarios', []))
        if not np.isclose(prob_sum, 100):
            warnings.append(f"In '{bucket_name}', the exit scenario probabilities sum to {prob_sum}%, but should be 100%.")
            
    return warnings


def load_predefined_models():
    """Loads fund model configurations from the 'models' directory."""
    models = []
    models_dir = "models"
    if os.path.isdir(models_dir):
        for filename in os.listdir(models_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(models_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        models.append(json.load(f))
                except Exception as e:
                    st.error(f"Error loading model {filename}: {e}")
    return models


def render_fund_model():
    """
    Acts as the entry point for the VC Fund Model page.
    It handles the initial choice of starting fresh or loading a file.
    """
    st.title("VC Fund Model Setup")

    if 'fund_model' not in st.session_state:
        st.info("Choose an option to begin.")

        # --- Predefined Models ---
        st.subheader("Start from a Predefined Model")
        predefined_models = load_predefined_models()

        if not predefined_models:
            st.warning("No predefined models found in the 'models' directory.")
        else:
            # Inject custom CSS to style the model cards
            st.markdown("""
                <style>
                    div[data-testid="stVerticalBlockBorderWrapper"] {
                        background-color: #FFF3E0;
                        border-radius: 0.5rem;
                    }
                </style>
                """, unsafe_allow_html=True)

            # Create a more robust and visually appealing card layout
            num_models = len(predefined_models)
            # Use a max of 3 columns for a clean grid layout
            cols = st.columns(min(num_models, 3))
            for i, model_data in enumerate(predefined_models):
                with cols[i % min(num_models, 3)]:
                    with st.container(border=True):
                        st.subheader(f"📄 {model_data.get('display_name', 'Unnamed Model')}")
                        st.markdown("---") # Visual separator

                        # Display key metrics
                        c1, c2 = st.columns(2)
                        c1.metric(label="Fund Size", value=f"${model_data.get('fund_size', 'N/A')}M")
                        c2.metric(label="Buckets", value=len(model_data.get('buckets', {})))

                        st.markdown("<br>", unsafe_allow_html=True) # Spacer

                        if st.button("Select This Model", key=f"select_model_{i}", use_container_width=True, type="secondary"):
                            st.session_state.fund_model = model_data
                            st.rerun()
        
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "Or Load Your Own Model from a JSON File",
            type=['json']
        )
        if uploaded_file is not None:
            try:
                loaded_data = json.load(uploaded_file)
                st.session_state.fund_model = loaded_data
                st.success("Model loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading or parsing file: {e}")
    else:
        render_fund_model_ui()

        if st.sidebar.button("Reset and Start Over"):
            del st.session_state.fund_model
            if 'simulation_results' in st.session_state:
                del st.session_state.simulation_results
            st.rerun() 