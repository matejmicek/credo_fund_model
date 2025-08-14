import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go
from scipy.stats import beta
import json
from copy import deepcopy
import os
import math
import re
import hashlib
from datetime import datetime

# Allow larger dataframes to be styled
pd.set_option("styler.render.max_elements", 1_000_000) # Set a large number instead of None

# --- Main Application Logic ---

def update_model_value(key_path, widget_key):
    """
    Generic callback to update a value in the nested fund_model dictionary.
    key_path is a list of keys to navigate the dictionary. Missing containers are created on the fly.
    """
    model = st.session_state.fund_model

    # Walk down the path and create intermediate containers as needed
    target = model
    for depth, key in enumerate(key_path[:-1]):
        next_key = key_path[depth + 1]

        if isinstance(target, dict):
            if key not in target:
                # Create appropriate container based on the next key type
                target[key] = [] if isinstance(next_key, int) else {}
            target = target[key]
        elif isinstance(target, list):
            # Ensure list is long enough
            if not isinstance(key, int):
                raise KeyError(f"Expected list index at depth {depth}, got {key!r}")
            while len(target) <= key:
                target.append({})
            target = target[key]
        else:
            # Unexpected structure; reset to dict to avoid crashing
            replacement = [] if isinstance(next_key, int) else {}
            target = replacement

    # Set the final value
    new_value = st.session_state[widget_key]

    # Handle dual-key slider case (expects last component to be [key_min, key_max])
    if isinstance(new_value, tuple) and len(key_path[-1]) == 2:
        last_min_key, last_max_key = key_path[-1]
        if isinstance(target, dict):
            target[last_min_key], target[last_max_key] = new_value
        else:
            raise KeyError("Cannot set tuple value on non-dict target")
    else:
        last_key = key_path[-1]
        if isinstance(target, dict):
            target[last_key] = new_value
        elif isinstance(target, list) and isinstance(last_key, int):
            while len(target) <= last_key:
                target.append(None)
            target[last_key] = new_value
        else:
            raise KeyError("Invalid target structure for final assignment")

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
        'follow_on_dilution_pct': 15,
        # New: multiple follow-on strategies (default one pre-populated)
        'follow_on_strategies': [
            {
                'name': 'Default',
                'probability': 50,
                'timing': 2.0,
                'size_pct_of_initial': 200,
                'valuation_multiple': 2.0,
                'dilution_pct': 15,
                'success_odds_factor': 1.0,
            }
        ],
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


def add_follow_on_strategy(bucket_key):
    """Add a new follow-on strategy to a bucket."""
    model = st.session_state.fund_model
    strategies = model['buckets'][bucket_key].setdefault('follow_on_strategies', [])
    strategies.append({
        'name': f"Strategy {len(strategies)+1}",
        'probability': 10.0,
        'timing': 2.0,
        'size_pct_of_initial': 100,
        'valuation_multiple': 2.0,
        'dilution_pct': 15.0,
        'success_odds_factor': 1.0,
    })
    if 'simulation_results' in st.session_state:
        del st.session_state.simulation_results


def remove_follow_on_strategy(bucket_key, strategy_index):
    """Remove a follow-on strategy from a bucket."""
    model = st.session_state.fund_model
    strategies = model['buckets'][bucket_key].get('follow_on_strategies', [])
    if 0 <= strategy_index < len(strategies):
        del strategies[strategy_index]
    if 'simulation_results' in st.session_state:
        del st.session_state.simulation_results

def apply_exit_model_probabilities():
    """
    Applies the probabilities from the Exit Probability Model to the current fund model.
    """
    if 'exit_model_probabilities' not in st.session_state:
        st.error("No probabilities from the Exit Model found.")
        return

    model = st.session_state.fund_model
    exit_probs = st.session_state['exit_model_probabilities']

    bucket_mapping = {
        "Pre-Seed Entry": "Pre-Seed",
        "Seed Entry": "Seed",
        "Large Pre-seed": "Large Pre-seed" 
    }

    for prob_name, bucket_name in bucket_mapping.items():
        # Find the bucket in the fund model that matches the name
        target_bucket_key = None
        for key, bucket_data in model.get('buckets', {}).items():
            if bucket_data.get('name') == bucket_name:
                target_bucket_key = key
                break
        
        if target_bucket_key:
            # Create new scenarios based on the exit model probabilities
            new_scenarios = []
            for exit_cat, prob in exit_probs[prob_name].items():
                if prob > 0:
                    val_str = exit_cat.split(' ')[0]
                    if val_str == '0':
                        min_val, max_val = 0, 0
                    elif 'M' in val_str:
                        min_val = max_val = float(val_str.replace('M', ''))
                    elif 'B' in val_str:
                        min_val = max_val = float(val_str.replace('B', '')) * 1000
                    else: # Fallback for unexpected format
                        min_val, max_val = 0, 0
                    
                    new_scenarios.append({
                        'name': f'Exit at {exit_cat}',
                        'probability': prob * 100,
                        'exit_valuation_min': min_val,
                        'exit_valuation_max': max_val,
                        'exit_year_min': 5, 'exit_year_max': 8, # Default values
                        'exit_dilution_pct': 20 # Default value
                    })
            
            # Replace the old scenarios with the new ones
            model['buckets'][target_bucket_key]['scenarios'] = new_scenarios
            st.success(f"Applied new exit probabilities to the '{bucket_name}' bucket.")

    # Invalidate simulation results
    if 'simulation_results' in st.session_state:
        del st.session_state.simulation_results


def render_fund_model_ui():
    """Renders the main UI once the model data is loaded into session state."""
    model = st.session_state.fund_model
    
    # Internal toggle for LP view (hide when forced via URL)
    forced_lp_mode = _get_query_params().get('view', [''])[0] == 'lp'
    if not forced_lp_mode:
        top_cols = st.columns([1, 1, 6])
        with top_cols[0]:
            lp_toggle = st.toggle("LP View", value=False, help="Switch to the LP presentation view")
        if lp_toggle:
            # Compute default slug from display name and navigate to LP URL
            default_slug = _slugify(model.get('display_name', 'model'))
            params = _get_query_params()
            params['view'] = 'lp'
            params['model'] = default_slug
            _set_query_params(params)
            st.rerun()
    st.title(f"🔮 {model.get('display_name', 'Probabilistic VC Fund Model')}")
    st.text_area(
        "Model Description",
        value=model.get('description', ''),
        key='fm_description',
        on_change=update_model_value,
        args=(['description'], 'fm_description'),
        help="A brief description of this fund model's strategy or purpose.",
        placeholder="e.g., A balanced fund targeting early-stage B2B SaaS companies with a focus on strong product-market fit."
    )
    # --- Integration with Exit Probability Model ---
    if 'exit_model_probabilities' in st.session_state:
        st.info("New exit probabilities from the Exit Probability Model are available.")
        if st.button("Apply to Current Fund Model", key="apply_exit_probs"):
            apply_exit_model_probabilities()

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
    
    # --- Sidebar: Allocation Summary (progress bars) ---
    try:
        fund_size_sb = float(model.get('fund_size', 0))
        follow_on_reserve_pct_sb = float(model.get('follow_on_reserve', 0))
        management_fee_reserve_sb = fund_size_sb * 0.17
        investable_capital_sb = fund_size_sb - management_fee_reserve_sb
        initial_capital_pool_sb = investable_capital_sb * (1 - follow_on_reserve_pct_sb / 100.0)
        follow_on_pool_sb = investable_capital_sb * (follow_on_reserve_pct_sb / 100.0)

        # Bucket coverage (how much of each pool is allocated across buckets)
        total_init_bucket_pct = sum(float(b.get('percentage', 0)) for b in model.get('buckets', {}).values())
        total_follow_bucket_pct = sum(float(b.get('follow_on_allocation_pct', 0)) for b in model.get('buckets', {}).values())

        init_alloc_ratio = max(0.0, min(1.0, total_init_bucket_pct / 100.0))
        fo_alloc_ratio = max(0.0, min(1.0, total_follow_bucket_pct / 100.0))

        st.sidebar.subheader("Allocation Summary")

        # Pure percentage labels (always render even if >100%)
        init_text = f"Initial allocation: {total_init_bucket_pct:.0f}%"
        fo_text = f"Follow-on allocation: {total_follow_bucket_pct:.0f}%"

        try:
            st.sidebar.progress(init_alloc_ratio, text=init_text)
        except TypeError:
            st.sidebar.progress(init_alloc_ratio)
            st.sidebar.caption(init_text)

        try:
            st.sidebar.progress(fo_alloc_ratio, text=fo_text)
        except TypeError:
            st.sidebar.progress(fo_alloc_ratio)
            st.sidebar.caption(fo_text)

        if total_init_bucket_pct > 100 or total_follow_bucket_pct > 100:
            st.sidebar.warning("Bucket allocations exceed 100%. Please rebalance.")
    except Exception:
        # Fail silently in sidebar summary if any field is temporarily invalid
        pass
    
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
            st.subheader("Follow-on Strategies")
            st.caption("Configure one or more mutually exclusive follow-on strategies. The sum of their probabilities should not exceed 100%.")

            # Render strategies
            strategies = bucket.get('follow_on_strategies', [])
            # Backwards compatibility: if no strategies present, create a legacy-equivalent one on the fly
            if not strategies:
                strategies = [{
                    'name': 'Legacy',
                    'probability': float(bucket.get('follow_on_probability', 0.0)),
                    'timing': float(bucket.get('follow_on_timing', 2.0)),
                    'size_pct_of_initial': float(bucket.get('follow_on_size_pct_of_initial', 0.0)),
                    'valuation_multiple': float(bucket.get('follow_on_valuation_multiple', 2.0)),
                    'dilution_pct': float(bucket.get('follow_on_dilution_pct', 15.0)),
                }]
            total_strategy_prob = sum(float(s.get('probability', 0)) for s in strategies)
            if total_strategy_prob > 100 + 1e-6:
                st.warning(f"Follow-on strategies sum to {total_strategy_prob:.1f}%, which exceeds 100%.")

            for s_idx, strat in enumerate(strategies):
                with st.container():
                    sc1, sc2, sc3, sc4, sc5, sc6, sc7 = st.columns([2, 2, 2, 2, 2, 2, 1])
                    with sc1:
                        st.text_input("Name", value=strat.get('name', f'Strategy {s_idx+1}'),
                                      key=f'fm_b_{i_str}_fo_{s_idx}_name', label_visibility="collapsed",
                                      on_change=update_model_value, args=(['buckets', i_str, 'follow_on_strategies', s_idx, 'name'], f'fm_b_{i_str}_fo_{s_idx}_name'))
                    with sc2:
                        st.number_input("Probability (%)", min_value=0.0, max_value=100.0, step=0.1, format="%.1f",
                                        value=float(strat.get('probability', 0.0)), key=f'fm_b_{i_str}_fo_{s_idx}_prob',
                                        on_change=update_model_value, args=(['buckets', i_str, 'follow_on_strategies', s_idx, 'probability'], f'fm_b_{i_str}_fo_{s_idx}_prob'))
                    with sc3:
                        st.number_input("Timing (yrs)", min_value=0.0, step=0.5, format="%.1f",
                                        value=float(strat.get('timing', 2.0)), key=f'fm_b_{i_str}_fo_{s_idx}_timing',
                                        on_change=update_model_value, args=(['buckets', i_str, 'follow_on_strategies', s_idx, 'timing'], f'fm_b_{i_str}_fo_{s_idx}_timing'))
                    with sc4:
                        st.number_input("Size (% init)", min_value=0, step=10,
                                        value=int(strat.get('size_pct_of_initial', 100)), key=f'fm_b_{i_str}_fo_{s_idx}_size',
                                        on_change=update_model_value, args=(['buckets', i_str, 'follow_on_strategies', s_idx, 'size_pct_of_initial'], f'fm_b_{i_str}_fo_{s_idx}_size'))
                    with sc5:
                        st.number_input("Valuation (x entry)", min_value=1.0, step=0.1, format="%.1f",
                                        value=float(strat.get('valuation_multiple', 2.0)), key=f'fm_b_{i_str}_fo_{s_idx}_val',
                                        on_change=update_model_value, args=(['buckets', i_str, 'follow_on_strategies', s_idx, 'valuation_multiple'], f'fm_b_{i_str}_fo_{s_idx}_val'))
                    with sc6:
                        st.number_input("Dilution (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.0f",
                                        value=float(strat.get('dilution_pct', 15.0)), key=f'fm_b_{i_str}_fo_{s_idx}_dil',
                                        on_change=update_model_value, args=(['buckets', i_str, 'follow_on_strategies', s_idx, 'dilution_pct'], f'fm_b_{i_str}_fo_{s_idx}_dil'))
                    with sc7:
                        st.number_input("Success Odds Factor", min_value=0.1, max_value=10.0, step=0.1, format="%.1f",
                                        value=float(strat.get('success_odds_factor', 1.0)), key=f'fm_b_{i_str}_fo_{s_idx}_sof',
                                        on_change=update_model_value, args=(['buckets', i_str, 'follow_on_strategies', s_idx, 'success_odds_factor'], f'fm_b_{i_str}_fo_{s_idx}_sof'))

                    st.button("🗑️ Remove", key=f'remove_fo_{i_str}_{s_idx}', use_container_width=False,
                              on_click=remove_follow_on_strategy, args=(i_str, s_idx))

                if s_idx < len(strategies) - 1:
                    st.markdown("---")

            st.button("Add Follow-on Strategy", key=f'add_fo_{i_str}', on_click=add_follow_on_strategy, args=(i_str,), use_container_width=True)

            # --- Dynamic Follow-on Calculation ---
            avg_ticket = bucket.get('avg_ticket', 0)
            strategies = bucket.get('follow_on_strategies', [])
            # Weighted expected follow-ons and capital need across strategies
            follow_on_prob_pct = sum(float(s.get('probability', 0.0)) for s in strategies)
            # For capital need, sum across strategies: expected deals * each strategy's avg ticket
            # Expected follow-on investments counts are based on total probability
            percentage_follow_on = bucket.get('follow_on_allocation_pct', 0)
            
            absolute_initial_for_bucket = initial_capital_pool * (bucket.get('percentage', 0) / 100)
            expected_initial_investments = (absolute_initial_for_bucket / avg_ticket) if avg_ticket > 0 else 0

            expected_follow_on_investments = expected_initial_investments * (follow_on_prob_pct / 100)
            needed_follow_on_capital = 0.0
            for s in strategies:
                s_prob = float(s.get('probability', 0.0)) / 100.0
                s_size_pct = float(s.get('size_pct_of_initial', 0.0)) / 100.0
                needed_follow_on_capital += expected_initial_investments * s_prob * (avg_ticket * s_size_pct)
            
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
        # --- Main simulation progress ---
        main_status_text = st.empty()
        main_progress_bar = st.progress(0)
        
        def _update_main_progress(ratio: float):
            # Clamp and render
            r = max(0.0, min(1.0, float(ratio)))
            try:
                main_progress_bar.progress(r, text=f"Main simulation: {int(r*100)}%")
            except TypeError:
                main_progress_bar.progress(r)
                main_status_text.text(f"Main simulation: {int(r*100)}%")

        with st.spinner("Running main simulation (1,000 iterations)... This might take a moment."):
            main_results_df = run_monte_carlo_simulation(st.session_state.fund_model, 1000, progress_cb=_update_main_progress)
            st.session_state.simulation_results = main_results_df
        # Clear main progress UI
        main_progress_bar.empty()
        main_status_text.empty()

        # --- Fund Size Sensitivity Analysis ---
        analysis_results = []
        base_fund_size = st.session_state.fund_model['fund_size']
        
        # Sensitivity deltas: -10, 0, +10, +20, +30 (in $M)
        deltas = list(range(-10, 31, 10))
        fund_sizes_to_test = sorted(set([
            s for s in (base_fund_size + np.array(deltas)).tolist() if s > 0
        ]))

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
                results_df = run_monte_carlo_simulation(model_copy, 500)

            # Calculate and store metrics for this fund size
            mean_tvpi = results_df['tvpi'].mean()
            median_tvpi = results_df['tvpi'].median()
            mean_moic = results_df['moic'].mean()
            median_moic = results_df['moic'].median()
            prob_3x = (results_df['tvpi'] >= 3).mean() * 100
            prob_5x = (results_df['tvpi'] >= 5).mean() * 100
            
            analysis_results.append({
                'fund_size': size,
                'mean_tvpi': mean_tvpi,
                'median_tvpi': median_tvpi,
                'mean_moic': mean_moic,
                'median_moic': median_moic,
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
        # Display aggregated runtime warnings once (avoid per-iteration slow I/O)
        warnings_list = st.session_state.get('simulation_runtime_warnings', [])
        if warnings_list:
            with st.expander("⚠️ Runtime Notes from Simulation", expanded=False):
                for w in warnings_list:
                    st.warning(w)

        st.markdown("---")
        st.header("📤 Publish LP Snapshot")
        st.caption("Create a read-only presentation for LPs with a shareable deep link.")

        # Slug input
        default_slug = _slugify(model.get('display_name', 'model'))
        slug = st.text_input("Model Slug", value=default_slug, help="Used in the deep link: ?view=lp&model=<slug>")

        # Check if artifacts exist
        exists = _published_artifacts_exist(slug)
        if exists:
            st.warning("Artifacts for this slug already exist. Publishing will overwrite them.")
            confirm = st.checkbox("I understand and confirm overwrite", value=False)
        else:
            confirm = True

        can_publish = confirm and ('simulation_results' in st.session_state)
        publish_clicked = st.button("Publish LP Snapshot", type="primary", disabled=not can_publish)
        if publish_clicked and can_publish:
            try:
                _publish_lp_snapshot(slug)
                share_link = _build_lp_link(slug)
                st.success("Published successfully!")
                st.markdown(f"Shareable link: {share_link}")
            except Exception as e:
                st.error(f"Failed to publish: {e}")


def run_monte_carlo_simulation(fund_model, num_simulations=10000, progress_cb=None):
    """
    Runs the Monte Carlo simulation for the VC fund model, including cash flow analysis for IRR.
    If provided, progress_cb is called with a float in [0,1] indicating completion progress.
    """
    FUND_LIFE_YEARS = 20  # Fund life for cash flow analysis
    FUND_LIFE_MONTHS = FUND_LIFE_YEARS * 12

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

    # Throttled runtime warning counters (avoid Streamlit I/O in hot loop)
    adjusted_success_odds_count = 0
    softened_strategy_rates_count = 0


    for sim_idx in range(num_simulations):
        cash_flows = np.zeros(FUND_LIFE_MONTHS)
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

        # Create a list of all initial investments with their deployment months
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
                # Random month within the selected deployment year
                month_offset = np.random.randint(0, 12)
                invest_month = year * 12 + month_offset
                all_investments.append({'bucket_key': i_str, 'bucket': bucket, 'invest_month': invest_month, 'invest_year': year})

        # Process each investment through its lifecycle
        for investment_idx, investment in enumerate(all_investments):
            bucket_key = investment['bucket_key']
            bucket = investment['bucket']
            invest_month = investment['invest_month']
            investment_year = investment['invest_year']
            avg_ticket = bucket.get('avg_ticket', 0)

            # Sample entry valuation for this specific investment
            entry_valuation_min = bucket.get('entry_valuation_min', 0)
            entry_valuation_max = bucket.get('entry_valuation_max', 0)
            entry_valuation = np.random.uniform(entry_valuation_min, entry_valuation_max) if entry_valuation_max > entry_valuation_min else entry_valuation_min

            # Track initial investment stats
            initial_capital_invested_by_bucket[bucket_key] += avg_ticket
            initial_investment_count_by_bucket[bucket_key] += 1

            # Initial investment cash flow
            cash_flows[invest_month] -= avg_ticket
            total_invested_cash += avg_ticket

            # --- Ownership and Return Calculation ---
            initial_ownership_pct = (avg_ticket / entry_valuation * 100) if entry_valuation > 0 else 0

            # Handle follow-on investment using mutually exclusive strategies
            follow_on_investment = 0.0
            follow_on_ownership_delta_pct = 0.0
            did_follow_on = False
            chosen_strategy = None

            strategies = bucket.get('follow_on_strategies', [])
            # Draw a uniform and pick a strategy if any triggers; ensure at most one
            rand = np.random.uniform(0, 100)
            cumulative = 0.0
            for s in strategies:
                cumulative += float(s.get('probability', 0.0))
                if rand < cumulative:
                    chosen_strategy = s
                    break

            if chosen_strategy is not None:
                # Check against the specific bucket's follow-on pool
                follow_on_pool_for_bucket = follow_on_sub_pools.get(bucket_key, 0)
                spent_from_pool = follow_on_capital_spent_by_bucket.get(bucket_key, 0)

                follow_on_size_pct = float(chosen_strategy.get('size_pct_of_initial', 0.0))
                follow_on_amount = avg_ticket * (follow_on_size_pct / 100.0)

                if spent_from_pool + follow_on_amount <= follow_on_pool_for_bucket:
                    follow_on_timing = float(chosen_strategy.get('timing', 2.0))
                    # Use ceiling to avoid placing the cash flow earlier than intended
                    follow_on_months = math.ceil(follow_on_timing * 12.0)
                    follow_on_month_index = min(FUND_LIFE_MONTHS - 1, invest_month + follow_on_months)

                    # Ensure month is within bounds for indexing
                    if follow_on_month_index < FUND_LIFE_MONTHS:
                        did_follow_on = True
                        follow_on_investment = follow_on_amount
                        follow_on_capital_spent_by_bucket[bucket_key] += follow_on_amount
                        follow_on_investment_count_by_bucket[bucket_key] += 1
                        cash_flows[follow_on_month_index] -= follow_on_investment
                        total_invested_cash += follow_on_investment

                        # Apply follow-on dilution to initial ownership, then add new ownership from follow-on
                        follow_on_val_multiple = float(chosen_strategy.get('valuation_multiple', bucket.get('follow_on_valuation_multiple', 2.0)))
                        follow_on_valuation = entry_valuation * follow_on_val_multiple
                        follow_on_dilution_pct = float(chosen_strategy.get('dilution_pct', bucket.get('follow_on_dilution_pct', 15)))
                        diluted_initial_ownership_pct = initial_ownership_pct * (1 - follow_on_dilution_pct / 100.0)
                        if follow_on_valuation > 0:
                            new_follow_on_ownership_pct = (follow_on_investment / follow_on_valuation * 100.0)
                        else:
                            new_follow_on_ownership_pct = 0.0
                        # Ownership delta relative to pre-follow-on initial
                        follow_on_ownership_delta_pct = (diluted_initial_ownership_pct + new_follow_on_ownership_pct) - initial_ownership_pct

            # Determine outcome
            scenarios = bucket.get('scenarios', [])
            if not scenarios: continue

            probs = np.array([s.get('probability', 0) for s in scenarios], dtype=float)
            if probs.sum() == 0: continue
            probs /= probs.sum()

            # Compute base success/failure split from scenarios
            success_indices = [idx for idx, s in enumerate(scenarios) if (s.get('exit_valuation_max', 0) or 0) > 0]
            failure_indices = [idx for idx in range(len(scenarios)) if idx not in success_indices]
            base_success_prob = probs[success_indices].sum() if success_indices else 0.0
            base_failure_prob = 1.0 - base_success_prob

            # Normalize compositions within success and failure groups
            success_comp = probs[success_indices] / base_success_prob if base_success_prob > 0 else np.array([])
            failure_comp = probs[failure_indices] / base_failure_prob if base_failure_prob > 0 else np.array([])

            # Determine strategy weights (groups): chosen strategy probability mass and residual no-follow
            strategies_for_success = bucket.get('follow_on_strategies', [])
            if not strategies_for_success:
                strategies_for_success = [{
                    'name': 'Legacy',
                    'probability': float(bucket.get('follow_on_probability', 0.0)),
                    'timing': float(bucket.get('follow_on_timing', 2.0)),
                    'size_pct_of_initial': float(bucket.get('follow_on_size_pct_of_initial', 0.0)),
                    'valuation_multiple': float(bucket.get('follow_on_valuation_multiple', 2.0)),
                    'dilution_pct': float(bucket.get('follow_on_dilution_pct', 15.0)),
                    'success_odds_factor': 1.0,
                }]
            w_strats = np.array([float(s.get('probability', 0.0)) for s in strategies_for_success]) / 100.0
            w_strats_sum = w_strats.sum()
            w_no = max(0.0, 1.0 - w_strats_sum)

            # Compute strategy success rates via odds scaling
            def odds(p):
                return p / (1 - p) if 0 < p < 1 else (np.inf if p >= 1 else 0.0)

            def prob_from_odds(o):
                return o / (1 + o) if o != np.inf else 1.0

            base_odds = odds(base_success_prob)
            strat_success_rates = []
            for s in strategies_for_success:
                factor = float(s.get('success_odds_factor', 1.0))
                o_g = base_odds * factor
                strat_success_rates.append(prob_from_odds(o_g))
            strat_success_rates = np.array(strat_success_rates)

            # Solve for no-follow success rate so that blended equals base_success_prob
            blended_strat_success = (w_strats * strat_success_rates).sum()
            s_no = None
            if w_no > 1e-9:
                s_no = (base_success_prob - blended_strat_success) / w_no
                # Clamp and warn if needed
                if s_no < 0 or s_no > 1:
                    s_no = min(max(s_no, 0.0), 1.0)
                    adjusted_success_odds_count += 1
            else:
                # No residual mass; reblend strategies to match base_success_prob by softening odds
                total_possible = blended_strat_success
                if not np.isclose(total_possible, base_success_prob):
                    # Scale toward base by linear interpolation in probability space
                    alpha = 0.0
                    if total_possible > 0:
                        alpha = min(1.0, base_success_prob / total_possible)
                    strat_success_rates = strat_success_rates * alpha + base_success_prob * (1 - alpha)
                    softened_strategy_rates_count += 1

            # Determine the active group's success rate for this company
            # Apply strategy odds boost only if a follow-on actually occurred
            if did_follow_on and chosen_strategy is not None:
                # Find its index in strategies_for_success by name match fallback to first match
                chosen_idx = 0
                for idx, s in enumerate(strategies_for_success):
                    if s is chosen_strategy:
                        chosen_idx = idx
                        break
                chosen_success_prob = float(strat_success_rates[chosen_idx]) if len(strat_success_rates) > 0 else base_success_prob
            else:
                chosen_success_prob = float(s_no if s_no is not None else base_success_prob)

            # Draw success/failure, then sample within that group by preserved composition
            is_success = np.random.uniform(0, 1) < chosen_success_prob
            if is_success and success_indices:
                pick = np.random.choice(len(success_indices), p=success_comp)
                chosen_scenario_index = success_indices[pick]
            else:
                # failure
                if failure_indices:
                    if len(failure_indices) == 1:
                        chosen_scenario_index = failure_indices[0]
                    else:
                        pick = np.random.choice(len(failure_indices), p=failure_comp)
                        chosen_scenario_index = failure_indices[pick]
                else:
                    # Edge case: no explicit failure scenario; fall back to base sampling
                    chosen_scenario_index = np.random.choice(len(scenarios), p=probs)
            chosen_scenario = scenarios[chosen_scenario_index]

            # Determine exit valuation
            exit_valuation = np.random.uniform(
                chosen_scenario.get('exit_valuation_min', 0.0),
                chosen_scenario.get('exit_valuation_max', 0.0)
            )
            # Scenario exit dilution (used whether or not exit occurs within fund life for recording)
            scenario_exit_dilution_pct = chosen_scenario.get('exit_dilution_pct', 20)
            
            # Ownership before exit (post follow-on mechanics if any)
            # Apply dilution from the follow-on round even if the fund could not participate due to pool constraints.
            if did_follow_on:
                ownership_before_exit_pct = initial_ownership_pct + follow_on_ownership_delta_pct
            elif chosen_strategy is not None:
                # Round occurred but we didn't participate
                chosen_follow_on_dilution_pct = float(chosen_strategy.get('dilution_pct', bucket.get('follow_on_dilution_pct', 15)))
                ownership_before_exit_pct = initial_ownership_pct * (1 - chosen_follow_on_dilution_pct / 100.0)
            else:
                ownership_before_exit_pct = initial_ownership_pct
            
            # Handle exit and realized value
            time_to_exit_months = np.random.randint(
                chosen_scenario.get('exit_year_min', 5) * 12, 
                chosen_scenario.get('exit_year_max', 8) * 12 + 1
            )
            exit_month = invest_month + time_to_exit_months

            realized_value = 0
            final_ownership_pct = 0
            status = "Active" # Default status

            if exit_month < FUND_LIFE_MONTHS:
                # Apply exit dilution
                final_ownership_pct = ownership_before_exit_pct * (1 - scenario_exit_dilution_pct / 100)
                
                realized_value = (final_ownership_pct / 100) * exit_valuation
                realized_value_by_bucket[bucket_key] += realized_value
                cash_flows[exit_month] += realized_value
                total_realized_value += realized_value
                
                status = "Exited" if exit_valuation > 0 else "Failed"

            # --- NEW: Store detailed company data ---
            portfolio_details.append({
                'company_id': f"Company {investment_idx + 1}",
                'investment_year': (invest_month // 12) + 1, # 1-based year for display
                'stage': bucket.get('name', 'N/A'),
                'initial_check': avg_ticket,
                'initial_ownership': initial_ownership_pct,
                'entry_valuation': entry_valuation,
                'follow_on': "Yes" if did_follow_on else "No",
                'follow_on_check': follow_on_investment,
                'ownership_after_follow_on': ownership_before_exit_pct,
                'follow_on_dilution_pct': (float(chosen_strategy.get('dilution_pct')) if (chosen_strategy is not None) else bucket.get('follow_on_dilution_pct', 15)),
                'ownership_after_follow_on_dilution': ((initial_ownership_pct * (1 - float(chosen_strategy.get('dilution_pct', 15)) / 100.0)) if (chosen_strategy is not None) else initial_ownership_pct),
                'ownership_from_follow_on': ((follow_on_investment / (entry_valuation * float(chosen_strategy.get('valuation_multiple', 2.0))) * 100.0) if (did_follow_on and chosen_strategy is not None and entry_valuation * float(chosen_strategy.get('valuation_multiple', 2.0)) > 0) else 0.0),
                'final_ownership_at_exit': final_ownership_pct,
                'status': status,
                'exit_year': ((exit_month // 12) + 1) if status != "Active" else None,
                'exit_valuation': exit_valuation if status != "Active" else None,
                'net_return': realized_value,
                'exit_scenario': chosen_scenario.get('name', 'N/A'),
                'exit_dilution_pct': (scenario_exit_dilution_pct if status != "Active" else None)
            })

        # Calculate metrics for the simulation run
        moic = total_realized_value / total_invested_cash if total_invested_cash > 0 else 0
        tvpi = total_realized_value / fund_size if fund_size > 0 else 0
        
        # --- IRR Calculations (Gross and Net) ---
        gross_irr = np.nan
        net_irr = np.nan
        
        try:
            # 1. Calculate Gross IRR (monthly), then annualize for reporting
            gross_irr_monthly = npf.irr(cash_flows)
            gross_irr = (1 + gross_irr_monthly) ** 12 - 1 if not np.isnan(gross_irr_monthly) else np.nan

            # 2. Calculate Net IRR (from LP's perspective with fees and carry), monthly then annualize
            lp_net_cash_flows = np.zeros(FUND_LIFE_MONTHS)
            total_contributions = 0
            lp_capital_returned = 0
            annual_fee = fund_size * 0.02 # 2% management fee
            total_fees_paid = 0.0
            max_total_fees = fund_size * 0.17 # Cap at 17%

            for month in range(FUND_LIFE_MONTHS):
                # Outflows for LP: Investments + Fees
                investment_in_month = cash_flows[month] if cash_flows[month] < 0 else 0
                
                fee_in_month = 0.0
                if month < 10 * 12 and total_fees_paid < max_total_fees:
                    fee_to_charge = min(annual_fee / 12.0, max_total_fees - total_fees_paid)
                    fee_in_month = -fee_to_charge
                    total_fees_paid += fee_to_charge
                
                lp_outflow = investment_in_month + fee_in_month
                lp_net_cash_flows[month] += lp_outflow
                total_contributions += -lp_outflow

                # Inflows for LP: Distributions from exits with waterfall logic
                distribution_in_month = cash_flows[month] if cash_flows[month] > 0 else 0
                if distribution_in_month > 0:
                    # First, return all contributed capital to LPs
                    capital_to_return_hurdle = total_contributions - lp_capital_returned
                    dist_for_capital_return = min(distribution_in_month, capital_to_return_hurdle)
                    
                    lp_net_cash_flows[month] += dist_for_capital_return
                    lp_capital_returned += dist_for_capital_return
                    
                    # Then, split remaining profit 80/20
                    profit_distribution = distribution_in_month - dist_for_capital_return
                    if profit_distribution > 0:
                        lp_share_of_profit = profit_distribution * 0.80 # 80% to LPs
                        lp_net_cash_flows[month] += lp_share_of_profit

            net_irr_monthly = npf.irr(lp_net_cash_flows)
            net_irr = (1 + net_irr_monthly) ** 12 - 1 if not np.isnan(net_irr_monthly) else np.nan

        except ValueError:
            # If IRR calculation fails for either, mark as NaN so downstream metrics exclude them
            gross_irr = np.nan
            net_irr = np.nan

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

        # Progress callback (throttled aggressively for performance)
        if progress_cb is not None:
            # Update at most ~50 times per run to minimize UI overhead
            throttle = max(1, num_simulations // 50)
            if (sim_idx + 1) % throttle == 0 or (sim_idx + 1) == num_simulations:
                try:
                    progress_cb((sim_idx + 1) / float(num_simulations))
                except Exception:
                    # Don't let UI failures break the simulation
                    pass

    # --- NEW: Return both simulation results and portfolio details ---
    results_df = pd.DataFrame(all_simulation_runs)
    results_df['portfolio_details'] = all_portfolios
    # Store aggregated runtime warnings for later display (single UI write)
    try:
        runtime_warnings = []
        if adjusted_success_odds_count > 0:
            runtime_warnings.append(
                f"Adjusted success odds to preserve scenario probabilities {adjusted_success_odds_count:,} times. Consider tuning Success Odds Factors or strategy probabilities."
            )
        if softened_strategy_rates_count > 0:
            runtime_warnings.append(
                f"Softened strategy success rates {softened_strategy_rates_count:,} times due to 100% strategy mass. Consider leaving some probability for no-follow."
            )
        st.session_state['simulation_runtime_warnings'] = runtime_warnings
    except Exception:
        pass
    
    return results_df


def display_simulation_results(results_df):
    """
    Displays the results of the Monte Carlo simulation.
    """
    st.header("📈 Simulation Results")

    # --- Metrics ---
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
    mean_tvpi = results_df['tvpi'].mean()
    median_tvpi = results_df['tvpi'].median()
    mean_moic = results_df['moic'].mean()
    median_moic = results_df['moic'].median()
    prob_3x = (results_df['tvpi'] >= 3).mean() * 100
    prob_5x = (results_df['tvpi'] >= 5).mean() * 100
    prob_loss = (results_df['tvpi'] < 1).mean() * 100

    col1.metric("Mean TVPI", f"{mean_tvpi:.2f}x")
    col2.metric("Median TVPI", f"{median_tvpi:.2f}x")
    col3.metric("Mean MOIC", f"{mean_moic:.2f}x")
    col4.metric("Median MOIC", f"{median_moic:.2f}x")
    col5.metric("P(Loss of Capital)", f"{prob_loss:.1f}%")
    col6.metric("P(TVPI > 3x)", f"{prob_3x:.1f}%")
    col7.metric("P(TVPI > 5x)", f"{prob_5x:.1f}%")

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
    
    # --- NEW: Add Example Portfolio Tab if data exists (interactive) or was published (LP) ---
    if 'portfolio_details' in results_df.columns or ('published_example_portfolios' in st.session_state):
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
        fig_tvpi.add_vline(x=median_tvpi, line_width=2, line_dash="dot", line_color="blue",
                      annotation_text=f"Median: {median_tvpi:.2f}x", annotation_position="top left")
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
            median_bucket_tvpi = bucket_tvpi.median()

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
            fig_bucket_tvpi.add_vline(
                x=median_bucket_tvpi,
                line_width=2,
                line_dash="dot",
                line_color="blue",
                annotation_text=f"Median: {median_bucket_tvpi:.2f}x",
                annotation_position="top left"
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
        median_moic = main_moic.median()
        fig_moic = go.Figure()
        fig_moic.add_trace(go.Histogram(x=main_moic, xbins=histogram_bins_moic, name='Distribution', histnorm='percent', marker_color='green'))
        fig_moic.add_vline(x=mean_moic, line_width=2, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {mean_moic:.2f}x", annotation_position="top right")
        fig_moic.add_vline(x=median_moic, line_width=2, line_dash="dot", line_color="blue",
                      annotation_text=f"Median: {median_moic:.2f}x", annotation_position="top left")
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
            median_bucket_moic = bucket_moic.median()

            fig_bucket_moic = go.Figure()
            fig_bucket_moic.add_trace(go.Histogram(x=bucket_moic, xbins=histogram_bins_moic, name='Distribution', histnorm='percent', marker_color='orange'))
            fig_bucket_moic.add_vline(x=mean_bucket_moic, line_width=2, line_dash="dash", line_color="red",
                                      annotation_text=f"Mean: {mean_bucket_moic:.2f}x", annotation_position="top right")
            fig_bucket_moic.add_vline(x=median_bucket_moic, line_width=2, line_dash="dot", line_color="blue",
                                      annotation_text=f"Median: {median_bucket_moic:.2f}x", annotation_position="top left")
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
        fig_irr.add_vline(x=median_gross_irr, line_width=2, line_dash="dot", line_color="blue",
                      annotation_text=f"Median: {median_gross_irr:.1f}%", annotation_position="top left")
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
        fig_irr_net.add_vline(x=median_net_irr, line_width=2, line_dash="dot", line_color="blue",
                      annotation_text=f"Median: {median_net_irr:.1f}%", annotation_position="top left")
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
            if 'portfolio_details' in results_df.columns:
                create_example_portfolios_tab(results_df, st.session_state.fund_model)
            elif 'published_example_portfolios' in st.session_state:
                create_example_portfolios_tab_from_published(st.session_state.published_example_portfolios, st.session_state.fund_model)


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
        'follow_on', 'follow_on_check',
        'follow_on_dilution_pct', 'ownership_after_follow_on_dilution', 'ownership_from_follow_on', 'ownership_after_follow_on',
        'final_ownership_at_exit', 'exit_dilution_pct', 'exit_year', 'exit_valuation', 'net_return'
    ]
    portfolio_df = portfolio_df[display_columns]

    # Format the DataFrame for better readability and hide the index
    styler = portfolio_df.style.format({
            'initial_check': "${:,.2f}M",
            'initial_ownership': "{:.2f}%",
            'entry_valuation': "${:,.1f}M",
            'follow_on_check': "${:,.2f}M",
            'follow_on_dilution_pct': "{:.0f}%",
            'ownership_after_follow_on_dilution': "{:.2f}%",
            'ownership_from_follow_on': "{:.2f}%",
            'ownership_after_follow_on': "{:.2f}%",
            'final_ownership_at_exit': "{:.2f}%",
            'exit_dilution_pct': "{:.0f}%",
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
    st.info("This analysis shows how key return metrics change based on the total fund size. The point corresponding to your configured fund size uses the 1k simulation, while other points use a faster 500 simulation.", icon="ℹ️")
    
    analysis_df = st.session_state.fund_size_analysis_results
    base_fund_size = st.session_state.fund_model['fund_size']

    # Chart 1: TVPI (Mean & Median) vs. Fund Size
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=analysis_df['fund_size'], y=analysis_df['mean_tvpi'], mode='lines+markers', name='Mean TVPI'))
    if 'median_tvpi' in analysis_df.columns:
        fig1.add_trace(go.Scatter(x=analysis_df['fund_size'], y=analysis_df['median_tvpi'], mode='lines+markers', name='Median TVPI'))
    # Highlight the base fund size
    base_point = analysis_df[analysis_df['fund_size'] == base_fund_size]
    if not base_point.empty:
        fig1.add_trace(go.Scatter(x=base_point['fund_size'], y=base_point['mean_tvpi'], mode='markers', marker=dict(color='red', size=10), name='Your Fund (Mean)'))
        if 'median_tvpi' in base_point.columns:
            fig1.add_trace(go.Scatter(x=base_point['fund_size'], y=base_point['median_tvpi'], mode='markers', marker=dict(color='blue', size=10), name='Your Fund (Median)'))
    fig1.add_hline(y=1, line_width=2, line_dash="dash", line_color="red")
    fig1.update_layout(
        title="TVPI vs. Fund Size (Mean & Median)",
        xaxis_title="Fund Size ($M)",
        yaxis_title="TVPI (x)",
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

    # Chart 4: MOIC (Mean & Median) vs. Fund Size
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=analysis_df['fund_size'], y=analysis_df['mean_moic'], mode='lines+markers', name='Mean MOIC'))
    if 'median_moic' in analysis_df.columns:
        fig4.add_trace(go.Scatter(x=analysis_df['fund_size'], y=analysis_df['median_moic'], mode='lines+markers', name='Median MOIC'))
    if not base_point.empty:
        fig4.add_trace(go.Scatter(x=base_point['fund_size'], y=base_point['mean_moic'], mode='markers', marker=dict(color='red', size=10), name='Your Fund (Mean)'))
        if 'median_moic' in base_point.columns:
            fig4.add_trace(go.Scatter(x=base_point['fund_size'], y=base_point['median_moic'], mode='markers', marker=dict(color='blue', size=10), name='Your Fund (Median)'))
    fig4.add_hline(y=1, line_width=2, line_dash="dash", line_color="red")
    fig4.update_layout(
        title="MOIC vs. Fund Size (Mean & Median)",
        xaxis_title="Fund Size ($M)",
        yaxis_title="MOIC (x)",
        yaxis_range=[0, None]
    )
    st.plotly_chart(fig4, use_container_width=True)


def create_example_portfolios_tab_from_published(published_data: dict, fund_model: dict):
    """
    Renders the Example Portfolios tab from precomputed portfolios stored during publish.
    Expected structure:
    {
      "p50": {"run_metrics": {...}, "portfolio_details": [...]},
      "p65": {"run_metrics": {...}, "portfolio_details": [...]},
      "p90": {"run_metrics": {...}, "portfolio_details": [...]}
    }
    """
    st.header("✨ Representative Portfolio Outcomes (Published)")
    st.info("These portfolios were preselected during publishing and do not require recomputation.", icon="ℹ️")

    label_map = {
        'p50': "Median Case (50th Percentile)",
        'p65': "Upper-Median Case (65th Percentile)",
        'p90': "Bull Case (90th Percentile)",
    }
    keys = ['p50', 'p65', 'p90']
    tabs = st.tabs([label_map[k] for k in keys])
    for idx, k in enumerate(keys):
        data = published_data.get(k, {})
        with tabs[idx]:
            _render_single_published_portfolio(data, fund_model, label_map[k])


def _render_single_published_portfolio(data: dict, fund_model: dict, tab_title: str):
    portfolio_details = data.get('portfolio_details', [])
    run_metrics = data.get('run_metrics', {})
    if not portfolio_details:
        st.warning("No portfolio data available for this case.")
        return

    portfolio_df = pd.DataFrame(portfolio_details)
    st.subheader("Portfolio Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Portfolio TVPI", f"{run_metrics.get('tvpi', float('nan')):.2f}x")
    col2.metric("Portfolio MOIC", f"{run_metrics.get('moic', float('nan')):.2f}x")
    col3.metric("Gross IRR", f"{run_metrics.get('gross_irr', float('nan')):.1%}")
    col4.metric("Net IRR", f"{run_metrics.get('net_irr', float('nan')):.1%}")

    # Deployment summary
    st.subheader("Deployment Summary")
    total_deployed = portfolio_df['initial_check'].sum() + portfolio_df['follow_on_check'].sum()
    total_proceeds = portfolio_df['net_return'].sum()
    sum_c1, sum_c2 = st.columns(2)
    sum_c1.metric("Total Capital Deployed", f"${total_deployed:.2f}M")
    sum_c2.metric("Total Proceeds", f"${total_proceeds:.2f}M")

    with st.expander("Show Deployment Statistics by Bucket for this Portfolio"):
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

        bucket_stats_df = portfolio_df.groupby('stage').agg(
            deployed_initial=('initial_check', 'sum'),
            count_initial=('company_id', 'size'),
            deployed_follow_on=('follow_on_check', 'sum'),
            count_follow_on=('follow_on', lambda x: (x == 'Yes').sum())
        ).reset_index()

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

    # Exit Distribution
    st.subheader("Exit Distribution")
    exited_df = portfolio_df[portfolio_df['status'].isin(['Exited', 'Failed'])].copy()
    scenario_map = {
        "Failure": "0", "$10M Exit": "10M", "$50M Exit": "50M",
        "$100M Exit": "100M", "$500M Exit": "500M", "$1B+ Exit": "1B",
        "Base Case": "Base", "Home Run": "Home Run"
    }
    ordered_scenarios = list(scenario_map.keys())
    exited_df['exit_category'] = pd.Categorical(
        exited_df['exit_scenario'], categories=ordered_scenarios, ordered=True
    )
    exit_counts = exited_df['exit_category'].value_counts().sort_index()
    chart_data = pd.DataFrame({'scenario': exit_counts.index, 'count': exit_counts.values})
    chart_data['label'] = chart_data['scenario'].map(scenario_map)
    if not chart_data.empty:
        fig = go.Figure(data=[
            go.Bar(x=chart_data['label'], y=chart_data['count'], text=chart_data['count'],
                   textposition='inside', marker_color='royalblue',
                   textfont=dict(color='white', size=14, family='Arial, sans-serif'))
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

    # Company table
    st.subheader("Company Investment Details")
    display_columns = [
        'company_id', 'investment_year', 'stage', 'status',
        'initial_check', 'initial_ownership',
        'follow_on', 'follow_on_check',
        'follow_on_dilution_pct', 'ownership_after_follow_on_dilution', 'ownership_from_follow_on', 'ownership_after_follow_on',
        'final_ownership_at_exit', 'exit_dilution_pct', 'exit_year', 'exit_valuation', 'net_return'
    ]
    portfolio_df = portfolio_df[display_columns]
    styler = portfolio_df.style.format({
        'initial_check': "${:,.2f}M",
        'initial_ownership': "{:.2f}%",
        'follow_on_check': "${:,.2f}M",
        'follow_on_dilution_pct': "{:.0f}%",
        'ownership_after_follow_on_dilution': "{:.2f}%",
        'ownership_from_follow_on': "{:.2f}%",
        'ownership_after_follow_on': "{:.2f}%",
        'final_ownership_at_exit': "{:.2f}%",
        'exit_dilution_pct': "{:.0f}%",
        'exit_valuation': "${:,.1f}M",
        'net_return': "${:,.2f}M",
    }).set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector='th', props=[('text-align', 'left')])]).hide(axis="index")
    table_height = (len(portfolio_df) + 1) * 35
    st.dataframe(styler, height=table_height)

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

    # LP deep link handling: if view=lp&model=slug present, render LP view and return
    params = _get_query_params()
    if params.get('view', [''])[0] == 'lp' and params.get('model', ['']):
        slug = params.get('model', [''])[0]
        _render_lp_view(slug)
        return

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


# ======================
# LP VIEW + PUBLISH UTIL
# ======================

def _slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9\-\s]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s or "model"


def _published_dir() -> str:
    d = os.path.join("models", "published")
    os.makedirs(d, exist_ok=True)
    return d


def _published_paths(slug: str) -> dict:
    base = os.path.join(_published_dir(), slug)
    return {
        'model_json': f"{base}.json",
        'results_parquet': f"{base}_results.parquet",
        'fund_size_parquet': f"{base}_fund_size.parquet",
        'portfolios_json': f"{base}_portfolios.json",
        'meta_json': f"{base}_meta.json",
    }


def _published_artifacts_exist(slug: str) -> bool:
    paths = _published_paths(slug)
    return any(os.path.exists(p) for p in paths.values())


def _publish_lp_snapshot(slug: str) -> None:
    if 'fund_model' not in st.session_state or 'simulation_results' not in st.session_state:
        raise RuntimeError("No model/results in session to publish.")

    model = st.session_state.fund_model
    results_df: pd.DataFrame = st.session_state.simulation_results
    fund_size_df: pd.DataFrame = st.session_state.get('fund_size_analysis_results', pd.DataFrame())

    paths = _published_paths(slug)

    # 1) Save model snapshot (ensure JSON-serializable types)
    with open(paths['model_json'], 'w') as f:
        json.dump(_to_jsonable(model), f, indent=2)

    # 2) Save results without heavy portfolio_details to parquet
    results_to_save = results_df.copy()
    if 'portfolio_details' in results_to_save.columns:
        results_to_save = results_to_save.drop(columns=['portfolio_details'])
    results_to_save.to_parquet(paths['results_parquet'], index=False)

    # 3) Save fund size analysis if present
    if not fund_size_df.empty:
        fund_size_df.to_parquet(paths['fund_size_parquet'], index=False)

    # 4) Precompute representative portfolios and save compact JSON
    example_data = _compute_example_portfolios(results_df)
    with open(paths['portfolios_json'], 'w') as f:
        json.dump(_to_jsonable(example_data), f)

    # 5) Save meta
    model_json_str = json.dumps(_to_jsonable(model), sort_keys=True)
    meta = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'model_sha256': hashlib.sha256(model_json_str.encode('utf-8')).hexdigest(),
        'slug': slug,
    }
    with open(paths['meta_json'], 'w') as f:
        json.dump(meta, f, indent=2)


def _compute_example_portfolios(results_df: pd.DataFrame) -> dict:
    data = {}
    def pick(p: float):
        target = results_df['tvpi'].quantile(p)
        row = results_df.iloc[(results_df['tvpi'] - target).abs().argsort()[0]]
        details = row['portfolio_details'] if 'portfolio_details' in results_df.columns else []
        return {
            'run_metrics': {
                'tvpi': float(row.get('tvpi', np.nan)),
                'moic': float(row.get('moic', np.nan)),
                'gross_irr': float(row.get('gross_irr', np.nan)),
                'net_irr': float(row.get('net_irr', np.nan)),
            },
            'portfolio_details': details,
        }
    data['p50'] = pick(0.50)
    data['p65'] = pick(0.65)
    data['p90'] = pick(0.90)
    return data


def _render_lp_view(slug: str) -> None:
    # Hide sidebar and Streamlit menu/footer for LP cleanliness
    _inject_lp_css()

    # Load artifacts
    try:
        model, results_df, fund_size_df, example_portfolios = _load_published(slug)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    # Prime session state
    st.session_state.fund_model = model
    if fund_size_df is not None and not fund_size_df.empty:
        st.session_state.fund_size_analysis_results = fund_size_df
    else:
        if 'fund_size_analysis_results' in st.session_state:
            del st.session_state['fund_size_analysis_results']

    # Keep published example portfolios available for display_simulation_results
    st.session_state.published_example_portfolios = example_portfolios

    # Read-only assumptions
    _render_fund_model_assumptions_readonly(model)

    # Results
    display_simulation_results(results_df)


def _load_published(slug: str):
    paths = _published_paths(slug)
    missing = []
    if not os.path.exists(paths['model_json']):
        missing.append(paths['model_json'])
    if not os.path.exists(paths['results_parquet']):
        missing.append(paths['results_parquet'])
    if missing:
        raise FileNotFoundError(f"Published artifacts not found for slug '{slug}'. Missing: {', '.join(missing)}")

    with open(paths['model_json'], 'r') as f:
        model = json.load(f)
    results_df = pd.read_parquet(paths['results_parquet'])
    fund_size_df = pd.read_parquet(paths['fund_size_parquet']) if os.path.exists(paths['fund_size_parquet']) else pd.DataFrame()
    example_portfolios = {}
    if os.path.exists(paths['portfolios_json']):
        with open(paths['portfolios_json'], 'r') as f:
            example_portfolios = json.load(f)
    return model, results_df, fund_size_df, example_portfolios


def _render_fund_model_assumptions_readonly(model: dict) -> None:
    st.title(f"🔮 {model.get('display_name', 'Probabilistic VC Fund Model')} — LP View")
    if model.get('description'):
        st.markdown(model['description'])

    # Global config cards
    fund_size = float(model.get('fund_size', 0))
    follow_on_reserve_pct = float(model.get('follow_on_reserve', 0))
    management_fee_reserve = fund_size * 0.17
    investable_capital = fund_size - management_fee_reserve
    initial_capital_pool = investable_capital * (1 - follow_on_reserve_pct / 100)
    total_follow_on_pool = investable_capital * (follow_on_reserve_pct / 100)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fund Size", f"${fund_size:.0f}M")
    c2.metric("Fee Reserve (Cap)", f"${management_fee_reserve:.1f}M")
    c3.metric("Investable Capital", f"${investable_capital:.1f}M")
    c4.metric("Follow-on Reserve", f"{follow_on_reserve_pct:.0f}%")

    st.markdown("---")
    st.header("Investment Buckets")
    sorted_bucket_keys = sorted(model.get('buckets', {}).keys(), key=int)
    bucket_keys = list(sorted_bucket_keys)
    if bucket_keys:
        # Aim to fit 5-6 buckets on a single row; fall back if fewer
        cols_per_row = min(6, max(1, len(bucket_keys)))
        for start in range(0, len(bucket_keys), cols_per_row):
            row_keys = bucket_keys[start:start + cols_per_row]
            cols = st.columns(len(row_keys))
            for col, i_str in zip(cols, row_keys):
                bucket = model['buckets'][i_str]
                with col:
                    # Wrap tile content to draw vertical divider via CSS
                    st.markdown("<div class='bucket-col'>", unsafe_allow_html=True)
                    with st.container(border=False):
                        # Compact header
                        st.markdown(f"<div style='font-size:0.95rem; font-weight:700'>{bucket.get('name', '')} <span style='font-weight:500'>( {int(bucket.get('percentage', 0))}% initial / {int(bucket.get('follow_on_allocation_pct', 0))}% follow-on )</span></div>", unsafe_allow_html=True)
                        cc1, cc2, cc3 = st.columns(3)
                        cc1.metric("Avg Ticket", f"${float(bucket.get('avg_ticket', 0)):.1f}M")
                        cc2.metric("Entry Val Min", f"${float(bucket.get('entry_valuation_min', 0)):.1f}M")
                        cc3.metric("Entry Val Max", f"${float(bucket.get('entry_valuation_max', 0)):.1f}M")
                        # Ownership range
                        avg_ticket = float(bucket.get('avg_ticket', 0))
                        min_entry_val = float(bucket.get('entry_valuation_min', 0))
                        max_entry_val = float(bucket.get('entry_valuation_max', 0))
                        min_ownership = (avg_ticket / max_entry_val * 100) if max_entry_val > 0 else 0
                        max_ownership = (avg_ticket / min_entry_val * 100) if min_entry_val > 0 else 0
                        st.caption(f"Expected ownership range: {min_ownership:.1f}% – {max_ownership:.1f}%")

                        st.markdown("<div class='bucket-section-title'><strong>Deployment Schedule</strong></div>", unsafe_allow_html=True)
                        ds = [bucket.get('deploy_y1', 0), bucket.get('deploy_y2', 0), bucket.get('deploy_y3', 0), bucket.get('deploy_y4', 0)]
                        ds_labels = ["Year 1", "Year 2", "Year 3", "Year 4"]
                        # Removed progress bar per request; show concise summary only
                        st.caption(", ".join(f"{l}: {v}%" for l, v in zip(ds_labels, ds)))

                        st.markdown("<div class='bucket-section-title'><strong>Follow-on Strategies</strong></div>", unsafe_allow_html=True)
                        strategies = bucket.get('follow_on_strategies', [])
                        if strategies:
                            for s in strategies:
                                st.markdown(f"- {s.get('name', 'Strategy')}: {float(s.get('probability', 0)):.0f}% prob, {float(s.get('timing', 0)):.1f} yrs, size {int(s.get('size_pct_of_initial', 0))}% of initial, val x{float(s.get('valuation_multiple', 0)):.1f}, dil {float(s.get('dilution_pct', 0)):.0f}%")
                        else:
                            st.caption("No follow-on strategies defined.")

                        # Exit Scenarios are intentionally omitted from tiles; see matrix below for cross-bucket view
                    st.markdown("</div>", unsafe_allow_html=True)

        # Matrix overview of exit scenarios across all buckets
        st.markdown("---")
        _render_exit_scenarios_overview_matrix(model)


def _inject_lp_css():
    st.markdown(
        """
        <style>
        /* Hide sidebar and menu/footer */
        div[data-testid="stSidebar"], #MainMenu, footer { display: none !important; }
        /* Card polish */
        /* Neutralize default card styling for bucket tiles */
        div[data-testid="stVerticalBlockBorderWrapper"] { background-color: transparent; border-radius: 0; padding: 0; }
        /* Tighter headings/metrics to fit 5-6 columns */
        div[data-testid="stVerticalBlockBorderWrapper"] h1,
        div[data-testid="stVerticalBlockBorderWrapper"] h2,
        div[data-testid="stVerticalBlockBorderWrapper"] h3,
        div[data-testid="stVerticalBlockBorderWrapper"] h4 { margin: 0.25rem 0 0.5rem 0; }
        div[data-testid="stVerticalBlockBorderWrapper"] h2, 
        div[data-testid="stVerticalBlockBorderWrapper"] h3 { font-size: 1.0rem; line-height: 1.2; }
        /* Compact metric component */
        div[data-testid="stMetricValue"] { font-size: 0.95rem !important; }
        div[data-testid="stMetricLabel"] { font-size: 0.75rem !important; }
        /* Compact captions */
        p, .stCaption { font-size: 0.85rem; }
        /* Align section headers across tiles */
        .bucket-section-title { min-height: 22px; display: flex; align-items: center; }
        .bucket-table-headers { margin-top: 0.25rem; }
        /* Pills for matrix cells */
        .pill { display:inline-block; padding:2px 8px; border-radius:9999px; font-size:0.8rem; margin:3px 6px 3px 0; }
        .pill-yrs { background:#EEF2FF; color:#334155; }
        .pill-dil { background:#FFF1F2; color:#7F1D1D; }
        /* Matrix row/cell spacing */
        .matrix-cell { padding: 8px 0 12px 0; }
        .matrix-label { padding: 10px 0; }
        .matrix-header-center { text-align: center; }
        /* Vertical divider between bucket columns */
        .bucket-col { border-right: 1px solid #E5E7EB; padding: 0 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _get_query_params() -> dict:
    # Normalize to dict[str, List[str]] regardless of Streamlit version
    try:
        qp = st.query_params
        params = {}
        for k in qp.keys():
            v = qp[k]
            if isinstance(v, list):
                params[k] = v
            else:
                params[k] = [v]
        return params
    except Exception:
        try:
            return st.experimental_get_query_params()
        except Exception:
            return {}


def _set_query_params(params: dict) -> None:
    try:
        st.query_params.clear()
        for k, v in params.items():
            st.query_params[k] = v
    except Exception:
        try:
            st.experimental_set_query_params(**params)
        except Exception:
            pass


def _build_lp_link(slug: str) -> str:
    # Render link in code style to avoid long base URL issues
    return f"`?view=lp&model={slug}`"


def _to_jsonable(obj):
    """Recursively convert numpy/pandas types into JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    try:
        # Handle plain NaN/inf floats
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
    except Exception:
        pass
    return obj


def _render_exit_scenarios_cards(scenarios: list) -> None:
    # Compact cards, aim to fit all in a single row (cap at 6 per row)
    max_per_row = 6
    for start in range(0, len(scenarios), max_per_row):
        row = scenarios[start:start + max_per_row]
        cols = st.columns(len(row))
        for idx, scenario in enumerate(row):
            with cols[idx]:
                with st.container(border=True):
                    st.markdown(f"**{scenario.get('name', 'Scenario')}**")
                    # Compact key figures (avoid big metric widgets)
                    prob = f"{float(scenario.get('probability', 0)):.0f}%"
                    dil = f"{float(scenario.get('exit_dilution_pct', 0)):.0f}%"
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption("Prob")
                        st.markdown(prob)
                    with c2:
                        st.caption("Exit Dilution")
                        st.markdown(dil)

                    # Valuation (no math formatting; omit $ in the value line)
                    vmin = f"{float(scenario.get('exit_valuation_min', 0)):.0f}"
                    vmax = f"{float(scenario.get('exit_valuation_max', 0)):.0f}"
                    st.caption("Valuation Range ($M)")
                    st.markdown(f"{vmin}–{vmax}")

                    # Time to exit
                    ymin = int(scenario.get('exit_year_min', 0))
                    ymax = int(scenario.get('exit_year_max', 0))
                    st.caption("Time to Exit (Years)")
                    st.markdown(f"{ymin}–{ymax}")


def _render_exit_scenarios_horizontal(scenarios: list, unique_suffix: str = "") -> None:
    # Render scenarios as a single horizontal normalized stacked bar for compact comparison
    # Enforce canonical order to align segments across buckets
    canonical_order = [
        "Failure", "$10M Exit", "$50M Exit", "$100M Exit", "$500M Exit", "$1B+ Exit",
        "Base Case", "Home Run"
    ]
    prob_by_name = {s.get('name', ''): float(s.get('probability', 0)) for s in scenarios}
    # Keep only present names but in canonical order; append any extra names deterministically
    present_canonical = [n for n in canonical_order if n in prob_by_name]
    extras = sorted([n for n in prob_by_name.keys() if n not in canonical_order])
    names = present_canonical + extras
    probs = [prob_by_name.get(n, 0.0) for n in names]
    total = sum(probs) or 1.0
    probs = [p / total * 100.0 for p in probs]

    fig = go.Figure()
    cum = 0.0
    colors = [
        "#d62728",  # Failure
        "#ff9896",  # $10M
        "#98df8a",  # $50M
        "#2ca02c",  # $100M
        "#1f77b4",  # $500M
        "#9467bd",  # $1B+
        "#8c564b",  # Base Case
        "#e377c2",  # Home Run
    ]
    for i, (name, p) in enumerate(zip(names, probs)):
        fig.add_trace(go.Bar(
            x=[p], y=[""], orientation='h', name=name,
            width=0.5,
            marker_color=colors[i % len(colors)],
            hovertemplate=f"{name}: {p:.0f}%<extra></extra>"
        ))
        cum += p

    fig.update_layout(
        barmode='stack',
        showlegend=False,
        height=90,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, 100], showgrid=False, ticksuffix='%'),
        yaxis=dict(showticklabels=False, showgrid=False)
    )
    key_suffix = unique_suffix or str(abs(hash(tuple(names))))
    st.plotly_chart(fig, use_container_width=True, key=f"exits_{key_suffix}")


def _render_exit_scenarios_table(scenarios: list) -> None:
    # Canonical order for row alignment across buckets
    canonical_order = [
        "Failure", "$10M Exit", "$50M Exit", "$100M Exit", "$500M Exit", "$1B+ Exit",
        "Base Case", "Home Run"
    ]
    # Build rows
    rows = []
    seen = set()
    for name in canonical_order:
        for s in scenarios:
            if s.get('name') == name:
                rows.append(s)
                seen.add(id(s))
                break
    # Append any extra scenarios not in canonical order
    rows.extend([s for s in scenarios if id(s) not in seen])

    # Render compact table
    headers = ["Scenario", "Prob", "Valuation ($M)", "Dilution", "Exit (yrs)"]
    c1, c2, c3, c4, c5 = st.columns([2, 1, 2, 1, 1])
    c1.caption(headers[0]); c2.caption(headers[1]); c3.caption(headers[2]); c4.caption(headers[3]); c5.caption(headers[4])
    for s in rows:
        name = s.get('name', '')
        prob = f"{float(s.get('probability', 0)):.0f}%"
        vmin = f"{float(s.get('exit_valuation_min', 0)):.0f}"
        vmax = f"{float(s.get('exit_valuation_max', 0)):.0f}"
        dil = f"{float(s.get('exit_dilution_pct', 0)):.0f}%"
        ymin = int(s.get('exit_year_min', 0))
        ymax = int(s.get('exit_year_max', 0))
        r1, r2, r3, r4, r5 = st.columns([2, 1, 2, 1, 1])
        r1.markdown(name)
        r2.markdown(prob)
        r3.markdown(f"{vmin}–{vmax}")
        r4.markdown(dil)
        r5.markdown(f"{ymin}–{ymax}")


def _render_exit_scenarios_overview_matrix(model: dict) -> None:
    st.subheader("Exit Scenarios Matrix")
    buckets = model.get('buckets', {})
    keys = sorted(buckets.keys(), key=int)

    # Canonical order template (used only to order if present); rows are derived from the union across buckets
    canonical_order = [
        "Failure", "$10M Exit", "$50M Exit", "$100M Exit", "$500M Exit", "$1B+ Exit",
        "Base Case", "Home Run"
    ]

    # Build a mapping for each bucket: scenario name -> properties and collect union of scenario names
    per_bucket = {}
    names_union = set()
    for k in keys:
        scenarios = buckets[k].get('scenarios', [])
        m = {s.get('name', ''): s for s in scenarios}
        per_bucket[k] = m
        names_union.update([s.get('name', '') for s in scenarios])

    # Determine row order from union, respecting canonical ordering for known names
    present_canonical = [n for n in canonical_order if n in names_union]
    extras = sorted([n for n in names_union if n not in canonical_order])
    scenario_rows = present_canonical + extras

    # Build valuation range map per scenario from the first bucket that defines it
    scenario_to_val = {}
    for k in keys:
        for s in buckets[k].get('scenarios', []):
            name = s.get('name', '')
            if name and name not in scenario_to_val:
                vmin = f"{float(s.get('exit_valuation_min', 0)):.0f}"
                vmax = f"{float(s.get('exit_valuation_max', 0)):.0f}"
                scenario_to_val[name] = f"{vmin}–{vmax}"

    # Header row: empty left corner + one column per bucket
    left_col_width = 2
    header_cols = st.columns([left_col_width] + [1 for _ in keys])
    header_cols[0].markdown("<div class='matrix-label'><strong>Exit Scenario (Valuation $M)</strong></div>", unsafe_allow_html=True)
    for idx, k in enumerate(keys, start=1):
        header_cols[idx].markdown(
            f"<div class='matrix-label matrix-header-center'><strong>{buckets[k].get('name','Bucket ' + k)}</strong></div>",
            unsafe_allow_html=True
        )

    # Render each scenario row
    for name in scenario_rows:
        row_cols = st.columns([left_col_width] + [1 for _ in keys])
        valtxt = scenario_to_val.get(name, "")
        row_cols[0].markdown(f"<div class='matrix-cell'>{name} <span style='color:#64748B'>({valtxt})</span></div>", unsafe_allow_html=True)
        for cidx, k in enumerate(keys, start=1):
            s = per_bucket[k].get(name)
            if not s:
                row_cols[cidx].markdown("<div class='matrix-cell' style='text-align:center; color:#94a3b8'>–</div>", unsafe_allow_html=True)
            else:
                prob = f"{float(s.get('probability', 0)):.0f}%"
                dil = f"{float(s.get('exit_dilution_pct', 0)):.0f}%"
                ymin = int(s.get('exit_year_min', 0))
                ymax = int(s.get('exit_year_max', 0))
                # Compact multiline cell with pills for nicer look and centered content
                row_cols[cidx].markdown(
                    "<div class='matrix-cell' style='text-align:center'>"
                    + f"<strong>{prob}</strong><br/>"
                    + f"<span class='pill pill-dil'>Dil {dil}</span>"
                    + f"<span class='pill pill-yrs'>{ymin}–{ymax} yrs</span>"
                    + "</div>",
                    unsafe_allow_html=True
                )