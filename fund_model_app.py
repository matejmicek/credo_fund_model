import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import beta
import json
from copy import deepcopy

# --- Main Application Logic ---

def get_default_model():
    """Returns the default configuration for a new fund model."""
    return {
        'fund_size': 100,
        'follow_on_reserve': 40,
        'num_buckets': 2,
        'buckets': {
            str(i): {
                'name': f'Bucket {i+1}',
                'percentage': 50,
                'deploy_y1': 50, 'deploy_y2': 25, 'deploy_y3': 15, 'deploy_y4': 10,
                'avg_ticket': 1.0, 'ownership': 15.0,
                'exit_year_min': 5, 'exit_year_max': 9,
                'fail_prob': 60, 'fail_mult': 0.0,
                'base_prob': 30, 'base_mult_min': 1.0, 'base_mult_max': 3.0,
                'home_run_prob': 10, 'home_run_mult_min': 3.0, 'home_run_mult_max': 20.0,
                'follow_on_amount': 2.0, 'follow_on_timing': 2
            } for i in range(2)
        }
    }

def update_model_value(key_path, widget_key):
    """
    Generic callback to update a value in the nested fund_model dictionary.
    key_path is a list of keys to navigate the dictionary, e.g., ['buckets', '0', 'percentage']
    """
    # Navigate to the target dictionary
    target = st.session_state.fund_model
    for key in key_path[:-1]:
        target = target[key]
    
    # Update the final value
    target[key_path[-1]] = st.session_state[widget_key]

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

    # --- Main Panel for Bucket Configuration ---
    st.header("Investment Bucket Configuration")

    total_percentage = sum(int(b.get('percentage', 0)) for b in model.get('buckets', {}).values())
    if total_percentage != 100:
        st.warning(f"Bucket allocations sum to {total_percentage}%. They must sum to 100%.")

    for i_str, bucket in model.get('buckets', {}).items():
        with st.expander(f"Bucket {int(i_str)+1}: {bucket.get('name', '')} ({int(bucket.get('percentage', 0))}%)", expanded=True):
            
            st.text_input("Bucket Name", value=bucket.get('name'), key=f'fm_b{i_str}_name',
                          on_change=update_model_value, args=(['buckets', i_str, 'name'], f'fm_b{i_str}_name'))
            
            st.slider("Percentage of Initial Capital (%)", 0, 100, value=int(bucket.get('percentage', 0)), key=f'fm_b{i_str}_perc',
                      on_change=update_model_value, args=(['buckets', i_str, 'percentage'], f'fm_b{i_str}_perc'))

            st.markdown("---")
            st.subheader("Deployment Schedule")
            c1, c2, c3, c4 = st.columns(4)
            c1.number_input("Year 1 (%)", 0, 100, value=bucket.get('deploy_y1'), key=f'fm_b{i_str}_d1',
                            on_change=update_model_value, args=(['buckets', i_str, 'deploy_y1'], f'fm_b{i_str}_d1'))
            c2.number_input("Year 2 (%)", 0, 100, value=bucket.get('deploy_y2'), key=f'fm_b{i_str}_d2',
                            on_change=update_model_value, args=(['buckets', i_str, 'deploy_y2'], f'fm_b{i_str}_d2'))
            c3.number_input("Year 3 (%)", 0, 100, value=bucket.get('deploy_y3'), key=f'fm_b{i_str}_d3',
                            on_change=update_model_value, args=(['buckets', i_str, 'deploy_y3'], f'fm_b{i_str}_d3'))
            c4.number_input("Year 4 (%)", 0, 100, value=bucket.get('deploy_y4'), key=f'fm_b{i_str}_d4',
                            on_change=update_model_value, args=(['buckets', i_str, 'deploy_y4'], f'fm_b{i_str}_d4'))

            st.markdown("---")
            st.subheader("Exit Scenarios & Timing")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.info("Define three possible outcomes for each investment.")
                st.number_input("Failure Probability (%)", 0, 100, value=bucket.get('fail_prob'), key=f'fm_b{i_str}_fprob',
                                on_change=update_model_value, args=(['buckets', i_str, 'fail_prob'], f'fm_b{i_str}_fprob'))
                st.number_input("Base Hit Probability (%)", 0, 100, value=bucket.get('base_prob'), key=f'fm_b{i_str}_bprob',
                                on_change=update_model_value, args=(['buckets', i_str, 'base_prob'], f'fm_b{i_str}_bprob'))
                st.number_input("Home Run Probability (%)", 0, 100, value=bucket.get('home_run_prob'), key=f'fm_b{i_str}_hprob',
                                on_change=update_model_value, args=(['buckets', i_str, 'home_run_prob'], f'fm_b{i_str}_hprob'))

            with c2:
                st.info("Define the distribution of exit timing.")
                min_val, max_val = st.slider(
                    "Time to Exit (Years)", 2, 10,
                    value=(bucket.get('exit_year_min'), bucket.get('exit_year_max')),
                    key=f'fm_b{i_str}_exit' 
                )
                # Callbacks on sliders with tuples are tricky, manual update is safer
                bucket['exit_year_min'], bucket['exit_year_max'] = min_val, max_val
    
    st.divider()
    st.header("📈 Run Simulation")
    if st.button("Run Monte Carlo Simulation", type="primary"):
        st.info("Simulation logic not yet implemented.")


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
            st.rerun() 