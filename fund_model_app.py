import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go
from scipy.stats import beta
import json
from copy import deepcopy

# --- Main Application Logic ---

def get_default_model():
    """Returns the default configuration for a new fund model, tailored for an Eastern European market."""
    return {
        'fund_size': 30,
        'follow_on_reserve': 50,
        'buckets': {
            '0': {
                'name': 'Pre-Seed / Seed', 'percentage': 70,
                'deploy_y1': 70, 'deploy_y2': 30, 'deploy_y3': 0, 'deploy_y4': 0,
                'avg_ticket': 0.25,
                'entry_valuation_min': 1.5, 'entry_valuation_max': 4.0,
                'follow_on_allocation_pct': 70,
                'follow_on_probability': 60,
                'follow_on_timing': 1.5,
                'follow_on_size_pct_of_initial': 200,
                'follow_on_valuation_multiple': 2.5,
                'scenarios': [
                    {'name': 'Failure', 'probability': 70, 'exit_valuation_min': 0.0, 'exit_valuation_max': 0.5, 'exit_year_min': 2, 'exit_year_max': 5, 'exit_dilution_pct': 50},
                    {'name': 'Base Case', 'probability': 25, 'exit_valuation_min': 5.0, 'exit_valuation_max': 20.0, 'exit_year_min': 4, 'exit_year_max': 8, 'exit_dilution_pct': 30},
                    {'name': 'Home Run', 'probability': 5, 'exit_valuation_min': 30.0, 'exit_valuation_max': 100.0, 'exit_year_min': 5, 'exit_year_max': 10, 'exit_dilution_pct': 25},
                ]
            },
            '1': {
                'name': 'Series A', 'percentage': 30,
                'deploy_y1': 20, 'deploy_y2': 50, 'deploy_y3': 30, 'deploy_y4': 0,
                'avg_ticket': 1.5,
                'entry_valuation_min': 8.0, 'entry_valuation_max': 20.0,
                'follow_on_allocation_pct': 30,
                'follow_on_probability': 70,
                'follow_on_timing': 2.0,
                'follow_on_size_pct_of_initial': 150,
                'follow_on_valuation_multiple': 2.0,
                'scenarios': [
                    {'name': 'Failure', 'probability': 40, 'exit_valuation_min': 0.0, 'exit_valuation_max': 3.0, 'exit_year_min': 3, 'exit_year_max': 6, 'exit_dilution_pct': 40},
                    {'name': 'Base Case', 'probability': 50, 'exit_valuation_min': 30.0, 'exit_valuation_max': 100.0, 'exit_year_min': 4, 'exit_year_max': 8, 'exit_dilution_pct': 25},
                    {'name': 'Home Run', 'probability': 10, 'exit_valuation_min': 100.0, 'exit_valuation_max': 400.0, 'exit_year_min': 5, 'exit_year_max': 9, 'exit_dilution_pct': 20},
                ]
            }
        }
    }

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

def remove_scenario(bucket_key, scenario_index):
    """Callback to remove a scenario from a bucket."""
    model = st.session_state.fund_model
    if scenario_index < len(model['buckets'][bucket_key].get('scenarios', [])):
        del model['buckets'][bucket_key]['scenarios'][scenario_index]

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

def remove_bucket(bucket_key):
    """Callback to remove a bucket by its key."""
    model = st.session_state.fund_model
    if bucket_key in model.get('buckets', {}):
        del model['buckets'][bucket_key]

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
    
    st.sidebar.divider()
    st.sidebar.header("Bucket Management")
    st.sidebar.button("Add New Bucket", on_click=add_bucket, use_container_width=True)
    
    # --- Main Panel for Bucket Configuration ---
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("Investment Bucket Configuration")
    with col2:
        st.button("Add New Bucket", on_click=add_bucket, use_container_width=True)

    # Sort keys to ensure consistent order
    sorted_bucket_keys = sorted(model.get('buckets', {}).keys(), key=int)

    for i_str in sorted_bucket_keys:
        bucket = model['buckets'][i_str]
        with st.expander(f"Bucket: {bucket.get('name', '')} ({int(bucket.get('percentage', 0))}%)", expanded=True):
            
            c1, c2 = st.columns([3, 1])
            with c1:
                st.text_input("Bucket Name", value=bucket.get('name'), key=f'fm_b_{i_str}_name', label_visibility="collapsed", on_change=update_model_value, args=(['buckets', i_str, 'name'], f'fm_b_{i_str}_name'))
            with c2:
                st.button("Delete", key=f'remove_b_{i_str}', on_click=remove_bucket, args=(i_str,), use_container_width=True)

            alloc_c1, alloc_c2 = st.columns(2)
            with alloc_c1:
                st.slider("Percentage of Initial Capital (%)", 0, 100, value=int(bucket.get('percentage', 0)), key=f'fm_b_{i_str}_perc', on_change=update_model_value, args=(['buckets', i_str, 'percentage'], f'fm_b_{i_str}_perc'))
            with alloc_c2:
                st.slider("Percentage of Follow-on Capital (%)", 0, 100, value=int(bucket.get('follow_on_allocation_pct', 0)), key=f'fm_b_{i_str}_follow_perc', on_change=update_model_value, args=(['buckets', i_str, 'follow_on_allocation_pct'], f'fm_b_{i_str}_follow_perc'))


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
                value=bucket.get('avg_ticket'), key=f'fm_b_{i_str}_ticket', on_change=update_model_value, args=(['buckets', i_str, 'avg_ticket'], f'fm_b_{i_str}_ticket'))

            t_c2.number_input("Min Entry Valuation ($M)", min_value=0.0, step=1.0, format="%.1f",
                value=bucket.get('entry_valuation_min', 0.0), key=f'fm_b_{i_str}_entry_val_min', on_change=update_model_value, args=(['buckets', i_str, 'entry_valuation_min'], f'fm_b_{i_str}_entry_val_min'))
            
            t_c3.number_input("Max Entry Valuation ($M)", min_value=0.0, step=1.0, format="%.1f",
                value=bucket.get('entry_valuation_max', 0.0), key=f'fm_b_{i_str}_entry_val_max', on_change=update_model_value, args=(['buckets', i_str, 'entry_valuation_max'], f'fm_b_{i_str}_entry_val_max'))
            
            # Calculated ownership
            avg_ticket = bucket.get('avg_ticket', 0)
            min_entry_val = bucket.get('entry_valuation_min', 0.0)
            max_entry_val = bucket.get('entry_valuation_max', 0.0)
            
            min_ownership = (avg_ticket / max_entry_val * 100) if max_entry_val > 0 else 0
            max_ownership = (avg_ticket / min_entry_val * 100) if min_entry_val > 0 else 0
            
            st.info(f"**Calculated Ownership Range:** {min_ownership:.1f}% - {max_ownership:.1f}%")

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
                value=bucket.get('follow_on_valuation_multiple', 2.0),
                key=f'fm_b_{i_str}_foval',
                on_change=update_model_value, args=(['buckets', i_str, 'follow_on_valuation_multiple'], f'fm_b_{i_str}_foval')
            )


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
                        st.number_input("Probability (%)", 0, 100, value=scenario.get('probability'), key=f'fm_b_{i_str}_s{s_idx}_prob', on_change=update_model_value, args=(['buckets', i_str, 'scenarios', s_idx, 'probability'], f'fm_b_{i_str}_s{s_idx}_prob'))

                    # --- ROW 2: Exit Details ---
                    r2c1, r2c2, r2c3, r2c4 = st.columns([3, 2, 4, 1])
                    with r2c1:
                        st.markdown("**Exit Valuation ($M)**")
                        ev_c1, ev_c2 = st.columns(2)
                        ev_c1.number_input("Min", min_value=0.0, step=0.1, format="%.1f",
                            value=scenario.get('exit_valuation_min', 1.0),
                            key=f'fm_b_{i_str}_s{s_idx}_ev_min', label_visibility="collapsed", on_change=update_model_value, args=(['buckets', i_str, 'scenarios', s_idx, 'exit_valuation_min'], f'fm_b_{i_str}_s{s_idx}_ev_min')
                        )
                        ev_c2.number_input("Max", min_value=0.0, step=0.1, format="%.1f",
                            value=scenario.get('exit_valuation_max', 2.0),
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
    
    st.divider()
    
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
        with st.spinner(f"Running 10,000 simulations... This may take a moment."):
            results_df = run_monte_carlo_simulation(st.session_state.fund_model, 10000)
            st.session_state.simulation_results = results_df
    
    if 'simulation_results' in st.session_state:
        display_simulation_results(st.session_state.simulation_results)


def run_monte_carlo_simulation(fund_model, num_simulations=10000):
    """
    Runs the Monte Carlo simulation for the VC fund model, including cash flow analysis for IRR.
    """
    FUND_LIFE_YEARS = 20  # Fund life for cash flow analysis

    fund_size = fund_model['fund_size']
    follow_on_reserve_pct = fund_model['follow_on_reserve']
    initial_capital_pool = fund_size * (1 - follow_on_reserve_pct / 100)
    total_follow_on_pool = fund_size * (follow_on_reserve_pct / 100)

    # Create bucket-specific follow-on pools
    follow_on_sub_pools = {
        i_str: total_follow_on_pool * (bucket.get('follow_on_allocation_pct', 0) / 100)
        for i_str, bucket in fund_model['buckets'].items()
    }
    
    simulation_runs = []

    for _ in range(num_simulations):
        cash_flows = np.zeros(FUND_LIFE_YEARS)
        total_invested_cash = 0
        total_realized_value = 0
        
        # Track spent capital from each sub-pool
        follow_on_capital_spent_by_bucket = {i_str: 0 for i_str in fund_model['buckets']}

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
        for investment in all_investments:
            bucket_key = investment['bucket_key']
            bucket = investment['bucket']
            investment_year = investment['year']
            avg_ticket = bucket.get('avg_ticket', 0)

            # Sample entry valuation for this specific investment
            entry_valuation_min = bucket.get('entry_valuation_min', 0)
            entry_valuation_max = bucket.get('entry_valuation_max', 0)
            entry_valuation = np.random.uniform(entry_valuation_min, entry_valuation_max) if entry_valuation_max > entry_valuation_min else entry_valuation_min

            # Initial investment cash flow
            cash_flows[investment_year] -= avg_ticket
            total_invested_cash += avg_ticket

            # --- Ownership and Return Calculation ---
            initial_ownership_pct = (avg_ticket / entry_valuation * 100) if entry_valuation > 0 else 0
            
            # Handle follow-on investment based on bucket-level strategy
            follow_on_investment = 0
            follow_on_ownership_pct = 0
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
                        follow_on_investment = follow_on_amount
                        follow_on_capital_spent_by_bucket[bucket_key] += follow_on_amount
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
            
            total_ownership_pct = initial_ownership_pct + follow_on_ownership_pct
            
            # Handle exit and realized value
            time_to_exit = np.random.randint(
                chosen_scenario.get('exit_year_min', 5), 
                chosen_scenario.get('exit_year_max', 9) + 1
            )
            exit_year = investment_year + time_to_exit

            if exit_year < FUND_LIFE_YEARS:
                # Apply exit dilution
                exit_dilution_pct = chosen_scenario.get('exit_dilution_pct', 20)
                final_ownership_pct = total_ownership_pct * (1 - exit_dilution_pct / 100)
                
                realized_value = (final_ownership_pct / 100) * exit_valuation
                cash_flows[exit_year] += realized_value
                total_realized_value += realized_value

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

        simulation_runs.append({
            'moic': moic, 'tvpi': tvpi, 
            'gross_irr': gross_irr, 'net_irr': net_irr,
            'total_invested': total_invested_cash, 'total_realized': total_realized_value
        })

    return pd.DataFrame(simulation_runs)


def display_simulation_results(results_df):
    """
    Displays the results of the Monte Carlo simulation.
    """
    st.header("📈 Simulation Results")

    # --- Metrics ---
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    mean_tvpi = results_df['tvpi'].mean()
    median_tvpi = results_df['tvpi'].median()
    prob_3x = (results_df['tvpi'] >= 3).mean() * 100
    prob_loss = (results_df['tvpi'] < 1).mean() * 100

    col1.metric("Mean TVPI", f"{mean_tvpi:.2f}x")
    col2.metric("Median TVPI", f"{median_tvpi:.2f}x")
    col3.metric("P(TVPI > 3x)", f"{prob_3x:.1f}%")
    col4.metric("P(Loss of Capital)", f"{prob_loss:.1f}%")

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

    # --- Charts ---
    tab1, tab2, tab3 = st.tabs(["TVPI Distribution", "Gross IRR Distribution", "Net IRR Distribution"])

    with tab1:
        st.subheader("Distribution of Fund Returns (TVPI)")
        fig_tvpi = go.Figure()
        fig_tvpi.add_trace(go.Histogram(x=results_df['tvpi'], nbinsx=50, name='Distribution', histnorm='percent'))
        fig_tvpi.add_vline(x=mean_tvpi, line_width=2, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {mean_tvpi:.2f}x", annotation_position="top right")
        fig_tvpi.update_layout(
            title="Distribution of Fund Return Multiples (Total Value / Fund Size)",
            xaxis_title="Fund Return Multiple (TVPI)", yaxis_title="Probability (%)", bargap=0.1
        )
        st.plotly_chart(fig_tvpi, use_container_width=True)

    with tab2:
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

    with tab3:
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


    with st.expander("View Raw Simulation Data"):
        st.dataframe(results_df.style.format({
            'moic': '{:.2f}x',
            'tvpi': '{:.2f}x',
            'gross_irr': '{:.2%}',
            'net_irr': '{:.2%}',
            'total_invested': '${:,.2f}M',
            'total_realized': '${:,.2f}M',
        }))


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
        if prob_sum != 100:
            warnings.append(f"In '{bucket_name}', the exit scenario probabilities sum to {prob_sum}%, but should be 100%.")
            
    return warnings


def render_fund_model():
    """
    Acts as the entry point for the VC Fund Model page.
    It handles the initial choice of starting fresh or loading a file.
    """
    st.title("VC Fund Model Setup")

    if 'fund_model' not in st.session_state:
        st.info("Choose an option to begin.")
        
        if st.button("Start with a New Model", type="primary"):
            st.session_state.fund_model = get_default_model()
            st.rerun()

        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "Load Model from JSON File",
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