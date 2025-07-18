import streamlit as st
import pandas as pd
import time

# Project-specific imports
from constants import TIER1_INVESTORS
from data_utils import FundingStage
from analysis_utils import (
    compare_graduation_rates, compare_funding_achievement, create_funding_auc_chart,
    analyze_round_size_vs_graduation, analyze_round_size_vs_achievement, analyze_time_to_next_round
)
from fund_model_app import render_fund_model

st.set_page_config(layout="wide")
st.title("🎓 Unified Startup Analysis")
st.markdown("This dashboard provides a dynamic view of startup success metrics based on Tier-1 VC funding.")

# --- App Navigation ---
st.sidebar.title("App Navigation")
app_mode = st.sidebar.radio(
    "Choose a section to explore",
    ["Startup Analysis", "VC Fund Model"]
)

if app_mode == "Startup Analysis":
    # --- Data Loading (with Caching) ---
    @st.cache_data
    def load_data(filepath: str) -> pd.DataFrame:
        """
        Loads the pre-processed dataset from a Parquet file.
        This function is cached to improve app performance.
        """
        try:
            df = pd.read_parquet(filepath)
            return df
        except FileNotFoundError:
            st.error(f"Fatal Error: The data file '{filepath}' was not found. Please run the data preparation script first.")
            st.info("To prepare the data, run this command in your terminal: `python prepare_data.py`")
            return None

    # Load data on app startup and show a spinner
    with st.spinner("Loading analysis data... this might take a moment."):
        df_analysis = load_data('crunchbase_ready.parquet')

    # If data loading fails, stop the app
    if df_analysis is None:
        st.stop()

    # --- Sidebar for Global Filters ---
    st.sidebar.header("🔬 Analysis Configuration")

    # --- Analysis Mode Selector ---
    analysis_mode = st.sidebar.radio(
        "Choose analysis mode:",
        ("Investor Tier", "Geography"),
        key="analysis_mode",
        help="Switch between comparing by investor type (Tier-1 vs. Non-Tier-1) or by region (US vs. Europe)."
    )
    
    # Define the filtering stage selector outside the conditional block
    # to prevent it from being re-created on every switch.
    filtering_stage = st.sidebar.selectbox(
        "Select the funding stage to analyze:",
        options=list(FundingStage),
        format_func=lambda x: x.value,
        index=0,  # Default to Seed
        help="This single selection controls the entire dashboard. It defines the stage for the Tier-1 comparison and for the round-size analysis.",
        key="filtering_stage"
    )

    # --- Main Panel for Dynamic Analysis ---

    st.header(f"Analysis based on investment at the `{filtering_stage.value}` stage")

    # Dynamically filter the dataset based on the sidebar selection
    raised_stage_flag_col = f'raised_{filtering_stage.name.lower()}'
    
    # --- Data for Progression Rate & Funding Achievement Analysis ---
    # For a fair comparison, we only look at companies that raised the filtering stage.
    companies_at_stage = df_analysis[df_analysis[raised_stage_flag_col] == True].copy()
    
    # --- Dynamic Data Splitting based on Analysis Mode ---
    if analysis_mode == "Investor Tier":
        tier1_flag_col = f'raised_{filtering_stage.name.lower()}_from_tier1'
        df1 = companies_at_stage[companies_at_stage[tier1_flag_col] == True].copy()
        df2 = companies_at_stage[companies_at_stage[tier1_flag_col] == False].copy()
        label1, label2 = "Tier-1 VC Backed", "Non-Tier-1 VC Backed"
        comparison_title = "Tier-1 vs. Non-Tier-1"
        
    else: # Geography
        df1 = companies_at_stage[companies_at_stage['region'] == "United States"].copy()
        df2 = companies_at_stage[companies_at_stage['region'] == "Europe"].copy()
        label1, label2 = "United States", "Europe"
        comparison_title = "US vs. Europe"


    st.markdown(f"""
    The dataset is now split into two groups for **{comparison_title}** analysis, based on companies that successfully raised a **{filtering_stage.value}**:
    - **{label1}:** **{len(df1):,}** companies.
    - **{label2}:** **{len(df2):,}** companies.
    """)

    st.divider()

    # --- Section 1: Progression Rate Analysis ---
    st.subheader("📈 Progression Rate Comparison")

    # --- UI for selecting the graduation stage ---
    all_stages = list(FundingStage)
    initial_stage_index = all_stages.index(filtering_stage)
    progression_options = all_stages[initial_stage_index + 1:]

    if not progression_options:
        st.warning(f"No further progression stages to analyze after {filtering_stage.value}.")
    else:
        progression_stage = st.selectbox(
            f"Select progression target stage (from {filtering_stage.value}):",
            options=progression_options,
            format_func=lambda x: x.value,
            index=0,
            help=f"The funding stage to which companies are attempting to 'progress'. We are measuring the % of companies from the '{filtering_stage.value}' cohort that successfully raise this selected stage."
        )

        with st.spinner("Updating progression rate analysis..."):
            # Run the comparison for the progression_stage
            fig1, stats1 = compare_graduation_rates(
                df1, df2,
                graduation_stage=progression_stage,
                label1=label1,
                label2=label2,
                title=f"Progression Rate from {filtering_stage.value} to {progression_stage.value} ({comparison_title})",
                min_sample_size=0
            )
            
            # Display the results
            st.plotly_chart(fig1, use_container_width=True)

            # Display summary statistics
            col1, col2 = st.columns(2)
            s1 = stats1[label1]
            r1 = (s1['graduated'] / s1['total'] * 100) if s1['total'] > 0 else 0
            col1.metric(
                label=f"{label1} Progression Rate (to {progression_stage.value})",
                value=f"{r1:.1f}%",
                help=f"{s1['graduated']:,} of {s1['total']:,} companies progressed."
            )

            s2 = stats1[label2]
            r2 = (s2['graduated'] / s2['total'] * 100) if s2['total'] > 0 else 0
            col2.metric(
                label=f"{label2} Progression Rate (to {progression_stage.value})",
                value=f"{r2:.1f}%",
                delta=f"{r1 - r2:.1f}% vs {label1}",
                help=f"{s2['graduated']:,} of {s2['total']:,} companies progressed."
            )

    st.divider()

    # --- Section 2: Funding Achievement Analysis ---
    st.subheader(f"💰 Funding Achievement Comparison ({comparison_title})")

    # Add a slider in the main panel for choosing the funding threshold
    funding_millions = st.slider(
        "Select the minimum total funding amount (in millions USD):",
        min_value=1,
        max_value=1000,
        value=50,
        step=1,
        format="$%dM",
        help="The minimum total funding amount a company must have raised to be considered 'achieved'."
    )
    funding_threshold = funding_millions * 1_000_000

    with st.spinner("Updating funding achievement analysis..."):
        # Run the comparison for the funding threshold
        fig2, stats2 = compare_funding_achievement(
            df1, df2,
            funding_threshold=funding_threshold,
            label1=label1,
            label2=label2,
            title=f"Companies Raising at Least ${funding_threshold:,.0f}"
        )

        # Display the results
        st.plotly_chart(fig2, use_container_width=True)

        # Display summary statistics
        col3, col4 = st.columns(2)
        s3 = stats2[label1]
        r3 = (s3['achieved'] / s3['total'] * 100) if s3['total'] > 0 else 0
        col3.metric(
            label=f"{label1} Achievement Rate",
            value=f"{r3:.2f}%",
            help=f"{s3['achieved']:,} of {s3['total']:,} companies achieved the funding threshold."
        )

        s4 = stats2[label2]
        r4 = (s4['achieved'] / s4['total'] * 100) if s4['total'] > 0 else 0
        col4.metric(
            label=f"{label2} Achievement Rate",
            value=f"{r4:.2f}%",
            delta=f"{r3 - r4:.2f}% vs {label1}",
            help=f"{s4['achieved']:,} of {s4['total']:,} companies achieved the funding threshold."
        )

    st.divider()

    # --- Section 3: Cumulative Funding Distribution ---
    st.subheader(f"📊 Cumulative Funding Distribution ({comparison_title})")

    use_log_scale = st.checkbox("Use logarithmic scale for Y-axis", help="Check this to better see the differences at lower percentages.")

    with st.spinner("Generating funding distribution curve..."):
        auc_fig = create_funding_auc_chart(
            df1, df2,
            label1=label1,
            label2=label2,
            title="Comparing Funding Trajectories",
            use_log_y=use_log_scale
        )
        st.plotly_chart(auc_fig, use_container_width=True)

    st.divider()

    # --- Section 4: Analysis by Round Size ---
    st.header(f"🔬 Analysis by Size of {filtering_stage.value} Round")

    st.markdown("This section investigates how the *size* of the selected funding round correlates with future success metrics.")

    # Filter the data for this analysis
    round_size_col = f"{filtering_stage.name.lower()}_round_size_usd"
    base_df_for_sizing = df_analysis.dropna(subset=[round_size_col]).copy()

    # Split into Tier-1 and Non-Tier-1
    if analysis_mode == "Investor Tier":
        tier1_col = f"raised_{filtering_stage.name.lower()}_from_tier1"
        df1_sizing = base_df_for_sizing[base_df_for_sizing[tier1_col] == True].copy()
        df2_sizing = base_df_for_sizing[base_df_for_sizing[tier1_col] == False].copy()
    else: # Geography
        df1_sizing = base_df_for_sizing[base_df_for_sizing['region'] == "United States"].copy()
        df2_sizing = base_df_for_sizing[base_df_for_sizing['region'] == "Europe"].copy()

    st.markdown(f"Analyzing **{len(base_df_for_sizing):,}** companies that raised a `{filtering_stage.value}` with a known round size. This includes **{len(df1_sizing):,}** {label1} companies and **{len(df2_sizing):,}** {label2} companies.")

    st.divider()

    # --- Sub-Section 4.1: Round Size vs. Graduation ---
    st.subheader(f"🎓 Graduation Probability by {filtering_stage.value} Size ({comparison_title})")

    # Graduation options should only be stages after the selected one
    all_stages = list(FundingStage)
    initial_stage_index = all_stages.index(filtering_stage)
    grad_opts = all_stages[initial_stage_index + 1:]

    if not grad_opts:
        st.warning(f"No further graduation stages to analyze after {filtering_stage.value}.")
    else:
        graduation_stage_for_sizing = st.selectbox(
            "Select the graduation stage:",
            options=grad_opts,
            format_func=lambda x: x.value,
            index=0,
            key="graduation_stage_for_sizing"
        )

        if graduation_stage_for_sizing:
            with st.spinner(f"Analyzing graduation by {filtering_stage.value} size..."):
                fig_grad_by_size = analyze_round_size_vs_graduation(
                    df1_sizing, df2_sizing,
                    filtering_stage, graduation_stage_for_sizing,
                    label1=label1,
                    label2=label2,
                )
                st.plotly_chart(fig_grad_by_size, use_container_width=True)

    st.divider()

    # --- Sub-Section 4.2: Round Size vs. Time to Next Round ---
    st.subheader(f"⏱️ Time to Next Round by {filtering_stage.value} Size ({comparison_title})")

    # This chart is now independent of the graduation stage selector above.
    # It automatically determines the next consecutive round.
    all_stages = list(FundingStage)
    initial_stage_index = all_stages.index(filtering_stage)
    if initial_stage_index + 1 < len(all_stages):
        next_stage = all_stages[initial_stage_index + 1]
        st.markdown(f"This chart shows the median time taken to raise a `{next_stage.value}` after the `{filtering_stage.value}`.")

        with st.spinner(f"Analyzing time to next round from {filtering_stage.value} size..."):
            fig_time_by_size = analyze_time_to_next_round(
                df1_sizing, df2_sizing,
                filtering_stage,
                label1=label1,
                label2=label2,
            )
            st.plotly_chart(fig_time_by_size, use_container_width=True)
    else:
        st.info(f"There is no subsequent funding stage after '{filtering_stage.value}' to analyze for time.")

    st.divider()

    # --- Sub-Section 4.3: Round Size vs. Achievement ---
    st.subheader(f"💰 Funding Achievement by {filtering_stage.value} Size ({comparison_title})")

    funding_millions_for_sizing = st.slider(
        "Select the minimum total funding to compare (in millions USD):",
        min_value=1,
        max_value=1000,
        value=50,
        step=1,
        format="$%dM",
        key="funding_millions_for_sizing"
    )
    funding_threshold_for_sizing = funding_millions_for_sizing * 1_000_000

    with st.spinner(f"Analyzing achievement by {filtering_stage.value} size..."):
        fig_achieve_by_size = analyze_round_size_vs_achievement(
            df1_sizing, df2_sizing,
            filtering_stage, funding_threshold_for_sizing,
            label1=label1,
            label2=label2,
        )
        st.plotly_chart(fig_achieve_by_size, use_container_width=True) 

elif app_mode == "VC Fund Model":
    render_fund_model() 