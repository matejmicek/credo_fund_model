import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from typing import List, Tuple, Dict, Any
import numpy as np

# Project-specific imports
from data_utils import FundingStage

# --- Data Processing Functions ---

# The Preprocessor class has been moved to prepare_data.py
# This script now only contains analysis and visualization functions.

# --- Analysis & Visualization Function ---

@st.cache_data
def compare_graduation_rates(
    df1: pd.DataFrame, df2: pd.DataFrame,
    graduation_stage: FundingStage,
    label1: str = "Dataset 1",
    label2: str = "Dataset 2",
    title: str = "Progression Rate Comparison",
    min_sample_size: int = 10,
    year_range: Tuple[int, int] = (1995, 2021)
) -> Tuple[go.Figure, Dict]:
    """Compares the percentage of companies in two datasets that progress to a specific funding stage."""
    
    def get_round_types(json_str: str) -> set:
        """Helper function to parse round types from the JSON string."""
        try:
            rounds = set()
            enriched_rounds = json.loads(json_str)
            for r in enriched_rounds:
                if isinstance(r, dict) and 'value' in r and isinstance(r['value'], str) and ' - ' in r['value']:
                    round_type = r['value'].split(' - ')[0]
                    if any(member.value == round_type for member in FundingStage):
                        rounds.add(round_type)
            return rounds
        except (json.JSONDecodeError, TypeError):
            return set()

    def calculate_rates_vectorized(df: pd.DataFrame) -> pd.DataFrame:
        """Optimized helper to calculate progression rates."""
        if df.empty:
            return pd.DataFrame(columns=['year', 'total_companies', 'graduated', 'rate'])
        
        # Work on a copy to avoid mutating the cached object
        df_copy = df.copy()
        
        # Vectorized application of the parsing function
        df_copy['stages_set'] = df_copy['funding_rounds_list_enriched'].apply(get_round_types)
        
        # Vectorized check for graduation
        df_copy['graduated'] = df_copy['stages_set'].apply(lambda s: graduation_stage.value in s)

        # Group by year and aggregate. The denominator is ALL companies in the cohort for that year.
        yearly_stats = df_copy.groupby('founding_year').agg(
            total_companies=('id', 'count'),
            graduated=('graduated', 'sum')
        ).reset_index()
        
        yearly_stats = yearly_stats[yearly_stats['total_companies'] >= min_sample_size]
        
        if not yearly_stats.empty:
            yearly_stats['rate'] = (yearly_stats['graduated'] / yearly_stats['total_companies']) * 100
        else:
            yearly_stats['rate'] = 0

        yearly_stats = yearly_stats.rename(columns={'founding_year': 'year'})
        
        # Filter by year range
        return yearly_stats[yearly_stats['year'].between(year_range[0], year_range[1])]

    df_res1 = calculate_rates_vectorized(df1)
    df_res2 = calculate_rates_vectorized(df2)
    
    fig = go.Figure()
    for df_res, label in [(df_res1, label1), (df_res2, label2)]:
        if not df_res.empty:
            fig.add_trace(go.Scatter(
                x=df_res['year'], y=df_res['rate'], mode='lines+markers',
                name=f'{label} (to {graduation_stage.value})',
                customdata=df_res[['total_companies', 'graduated']],
                hovertemplate=f'<b>{label}</b><br>Year: %{{x}}<br>Rate: %{{y:.1f}}%<br>Total Companies: %{{customdata[0]}}<extra></extra>'
            ))

    fig.update_layout(
        title={'text': f'{title}<br><sub>% of companies that raised a {graduation_stage.value}</sub>', 'x': 0.5},
        xaxis_title='Founding Year', yaxis_title=f'Progression Rate to {graduation_stage.value} (%)',
        hovermode='x unified', legend=dict(x=0.01, y=0.99)
    )
    
    total1 = df_res1['total_companies'].sum() if not df_res1.empty else 0
    graduated1 = df_res1['graduated'].sum() if not df_res1.empty else 0
    total2 = df_res2['total_companies'].sum() if not df_res2.empty else 0
    graduated2 = df_res2['graduated'].sum() if not df_res2.empty else 0

    stats = {
        label1: {'total': total1, 'graduated': graduated1},
        label2: {'total': total2, 'graduated': graduated2}
    }
    
    return fig, stats 

@st.cache_data
def compare_funding_achievement(
    df1: pd.DataFrame, 
    df2: pd.DataFrame,
    funding_threshold: float,
    label1: str = "Dataset 1",
    label2: str = "Dataset 2",
    title: str = "Funding Achievement Comparison"
) -> Tuple[go.Figure, Dict]:
    """
    Compares the percentage of companies in two datasets that have raised a minimum total funding amount.
    """
    
    def calculate_achievement(df: pd.DataFrame) -> Dict[str, Any]:
        """Helper to calculate funding achievement for a single DataFrame."""
        total_companies = len(df)
        if total_companies == 0:
            return {'total': 0, 'achieved': 0, 'rate': 0}
        
        # Ensure 'total_funding_usd' column exists and is numeric
        if 'total_funding_usd' not in df.columns:
            # Or handle it more gracefully
            raise KeyError("'total_funding_usd' column not found in the DataFrame.")
            
        df['total_funding_usd'] = pd.to_numeric(df['total_funding_usd'], errors='coerce')
        
        achieved_companies = df[df['total_funding_usd'] >= funding_threshold]
        count_achieved = len(achieved_companies)
        
        rate = (count_achieved / total_companies) * 100 if total_companies > 0 else 0
        
        return {'total': total_companies, 'achieved': count_achieved, 'rate': rate}

    stats1 = calculate_achievement(df1)
    stats2 = calculate_achievement(df2)
    
    # --- Visualization ---
    labels = [label1, label2]
    rates = [stats1['rate'], stats2['rate']]
    
    fig = go.Figure(go.Bar(
        x=labels,
        y=rates,
        text=[f'{r:.2f}%' for r in rates],
        textposition='auto',
        marker_color=['#1f77b4', '#ff7f0e']
    ))
    
    # Dynamically set the y-axis range
    max_rate = max(rates) if rates else 0
    yaxis_range = [0, max_rate * 1.2] if max_rate > 0 else [0, 10]

    fig.update_layout(
        title={
            'text': f'{title}<br><sub>% of companies raising at least ${funding_threshold:,.0f}</sub>',
            'x': 0.5
        },
        yaxis_title="Percentage of Companies (%)",
        xaxis_title="Company Group",
        yaxis=dict(range=yaxis_range)
    )

    stats = {
        label1: {'total': stats1['total'], 'achieved': stats1['achieved']},
        label2: {'total': stats2['total'], 'achieved': stats2['achieved']}
    }
    
    return fig, stats 

@st.cache_data
def create_funding_auc_chart(
    df1: pd.DataFrame, 
    df2: pd.DataFrame,
    label1: str = "Dataset 1",
    label2: str = "Dataset 2",
    title: str = "Funding Achievement Distribution",
    use_log_y: bool = False
) -> go.Figure:
    """
    Creates a cumulative distribution chart showing the percentage of companies
    that have raised at least a certain amount of funding.
    """
    
    thresholds = np.logspace(7, 9, num=50) # From 10^7 ($10M) to 10^9 ($1B)
    
    def calculate_cumulative_rates(df: pd.DataFrame) -> List[float]:
        """Helper to calculate the percentage of companies meeting each threshold."""
        total_companies = len(df)
        if total_companies == 0:
            return [0.0] * len(thresholds)
        
        df['total_funding_usd'] = pd.to_numeric(df['total_funding_usd'], errors='coerce')
        
        rates = []
        for t in thresholds:
            achieved_count = df[df['total_funding_usd'] >= t].shape[0]
            rates.append((achieved_count / total_companies) * 100)
        return rates

    rates1 = calculate_cumulative_rates(df1)
    rates2 = calculate_cumulative_rates(df2)
    
    # --- Visualization ---
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=thresholds, y=rates1, mode='lines', name=label1
    ))
    fig.add_trace(go.Scatter(
        x=thresholds, y=rates2, mode='lines', name=label2
    ))
    
    fig.update_layout(
        title={'text': title, 'x': 0.5},
        xaxis_title="Total Funding Raised (USD)",
        yaxis_title="% of Companies Raising At Least This Amount",
        xaxis_type='log',
        yaxis_type='log' if use_log_y else 'linear',
        hovermode='x unified'
    )
    
    return fig 

@st.cache_data
def analyze_round_size_vs_graduation(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    initial_stage: "FundingStage",
    graduation_stage: "FundingStage",
    label1: str = "Dataset 1",
    label2: str = "Dataset 2"
) -> go.Figure:
    """Analyzes graduation probability based on shared, readable buckets of initial funding round size."""

    round_size_col = f"{initial_stage.name.lower()}_round_size_usd"
    graduation_flag_col = f"raised_{graduation_stage.name.lower()}"

    # --- Unified Bucketing Logic ---
    combined_series = pd.concat([df1[round_size_col], df2[round_size_col]]).dropna()

    # --- Dynamic Bin Calculation ---
    target_size_per_bucket = 200
    total_companies = len(combined_series)
    if total_companies == 0:
        return go.Figure().update_layout(title="No data available for bucketing.")

    calculated_bins = max(1, int(total_companies / target_size_per_bucket))
    MIN_BINS, MAX_BINS = 5, 20
    bins = max(MIN_BINS, min(calculated_bins, MAX_BINS))

    if combined_series.nunique() < bins:
        bins = combined_series.nunique()
        if bins < 2:
            return go.Figure().update_layout(title="Not enough unique data points to create buckets.")

    try:
        _, edges = pd.qcut(combined_series, q=bins, duplicates='drop', retbins=True)
    except ValueError:
        return go.Figure().update_layout(title="Could not create buckets from data distribution.")

    def format_value(v):
        if v >= 1_000_000: return f"${v/1_000_000:,.1f}M"
        if v >= 1_000: return f"${v/1_000:,.0f}k"
        return f"${v:,.0f}"

    labels = [f"{format_value(edges[i])} - {format_value(edges[i+1])}" for i in range(len(edges) - 1)]

    def process_df(df: pd.DataFrame):
        df_filtered = df.dropna(subset=[round_size_col])
        if df_filtered.empty:
            return pd.DataFrame({'bucket': labels, 'rate': 0, 'sample_size': 0})

        df_filtered['bucket'] = pd.cut(df_filtered[round_size_col], bins=edges, labels=labels, include_lowest=True)
        stats = df_filtered.groupby('bucket', observed=False).agg(
            sample_size=(graduation_flag_col, 'count'),
            graduated=(graduation_flag_col, 'sum')
        ).reset_index()
        stats['rate'] = (stats['graduated'] / stats['sample_size']).fillna(0) * 100
        return stats

    stats1 = process_df(df1)
    stats2 = process_df(df2)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=stats1['bucket'], y=stats1['rate'], name=label1,
                          customdata=stats1['sample_size'],
                          hovertemplate='Bucket: %{x}<br>Rate: %{y:.1f}%<br>Sample Size: %{customdata}<extra></extra>'))
    fig.add_trace(go.Bar(x=stats2['bucket'], y=stats2['rate'], name=label2,
                          customdata=stats2['sample_size'],
                          hovertemplate='Bucket: %{x}<br>Rate: %{y:.1f}%<br>Sample Size: %{customdata}<extra></extra>'))

    fig.update_layout(
        title=f"Graduation from {initial_stage.value} by Round Size",
        xaxis_title=f"Size of {initial_stage.value} (USD) - Bucketed",
        yaxis_title=f"Probability of Raising {graduation_stage.value} (%)",
        hovermode='x unified',
        barmode='group'
    )
    return fig

@st.cache_data
def analyze_round_size_vs_achievement(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    initial_stage: "FundingStage",
    funding_threshold: float,
    label1: str = "Dataset 1",
    label2: str = "Dataset 2"
) -> go.Figure:
    """Analyzes total funding achievement based on the size of an initial funding round."""

    round_size_col = f"{initial_stage.name.lower()}_round_size_usd"

    # --- Unified Bucketing Logic ---
    combined_series = pd.concat([df1[round_size_col], df2[round_size_col]]).dropna()
    
    # --- Dynamic Bin Calculation ---
    target_size_per_bucket = 200
    total_companies = len(combined_series)
    if total_companies == 0:
        return go.Figure().update_layout(title="No data available for bucketing.")

    calculated_bins = max(1, int(total_companies / target_size_per_bucket))
    MIN_BINS, MAX_BINS = 5, 20
    bins = max(MIN_BINS, min(calculated_bins, MAX_BINS))

    if combined_series.nunique() < bins:
        bins = combined_series.nunique()
        if bins < 2:
            return go.Figure().update_layout(title="Not enough unique data points to create buckets.")

    try:
        _, edges = pd.qcut(combined_series, q=bins, duplicates='drop', retbins=True)
    except ValueError:
        return go.Figure().update_layout(title="Could not create buckets from data distribution.")

    def format_value(v):
        if v >= 1_000_000: return f"${v/1_000_000:,.1f}M"
        if v >= 1_000: return f"${v/1_000:,.0f}k"
        return f"${v:,.0f}"

    labels = [f"{format_value(edges[i])} - {format_value(edges[i+1])}" for i in range(len(edges) - 1)]

    def process_df(df: pd.DataFrame):
        df_filtered = df.dropna(subset=[round_size_col])
        if df_filtered.empty:
            return pd.DataFrame({'bucket': labels, 'rate': 0, 'sample_size': 0})
        
        df_filtered['bucket'] = pd.cut(df_filtered[round_size_col], bins=edges, labels=labels, include_lowest=True)

        df_filtered['achieved'] = df_filtered['total_funding_usd'] >= funding_threshold

        stats = df_filtered.groupby('bucket', observed=False).agg(
            sample_size=('achieved', 'count'),
            achieved_count=('achieved', 'sum')
        ).reset_index()

        stats['rate'] = (stats['achieved_count'] / stats['sample_size']).fillna(0) * 100
        return stats

    stats1 = process_df(df1)
    stats2 = process_df(df2)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=stats1['bucket'], y=stats1['rate'], name=label1,
                          customdata=stats1['sample_size'],
                          hovertemplate='Bucket: %{x}<br>Achievement Rate: %{y:.1f}%<br>Sample Size: %{customdata}<extra></extra>'))
    fig.add_trace(go.Bar(x=stats2['bucket'], y=stats2['rate'], name=label2,
                          customdata=stats2['sample_size'],
                          hovertemplate='Bucket: %{x}<br>Achievement Rate: %{y:.1f}%<br>Sample Size: %{customdata}<extra></extra>'))

    fig.update_layout(
        title=f"Funding Achievement by Size of {initial_stage.value}",
        xaxis_title=f"Size of {initial_stage.value} (USD) - Bucketed",
        yaxis_title=f"% of Companies Raising At Least ${funding_threshold:,.0f}",
        hovermode='x unified',
        barmode='group'
    )
    return fig 

@st.cache_data
def analyze_time_to_next_round(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    initial_stage: "FundingStage",
    label1: str = "Dataset 1",
    label2: str = "Dataset 2"
) -> go.Figure:
    """Analyzes the median time to the next funding round based on the size of the initial round."""

    all_stages = list(FundingStage)
    initial_stage_index = all_stages.index(initial_stage)

    if initial_stage_index + 1 >= len(all_stages):
        return go.Figure().update_layout(title=f"No subsequent round to analyze for time after {initial_stage.value}.")

    next_stage = all_stages[initial_stage_index + 1]
    
    round_size_col = f"{initial_stage.name.lower()}_round_size_usd"
    time_col = f"time_from_{initial_stage.name.lower()}_to_{next_stage.name.lower()}_days"
    
    # Defensively check if the time column exists before proceeding
    if time_col not in df1.columns or time_col not in df2.columns:
        return go.Figure().update_layout(
            title=f"Data Not Available",
            annotations=[dict(
                text=f"The required data column '{time_col}' was not found in the dataset.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )]
        )

    # --- Unified Bucketing Logic ---
    # Combine series from both dataframes that have a valid time-to-next-round
    combined_series = pd.concat([
        df1.dropna(subset=[time_col])[round_size_col],
        df2.dropna(subset=[time_col])[round_size_col]
    ]).dropna()

    # --- Dynamic Bin Calculation ---
    target_size_per_bucket = 100 # Smaller target for time analysis
    total_companies = len(combined_series)
    if total_companies == 0:
        return go.Figure().update_layout(title="No data available for bucketing.")

    calculated_bins = max(1, int(total_companies / target_size_per_bucket))
    MIN_BINS, MAX_BINS = 5, 15 # Adjusted max bins
    bins = max(MIN_BINS, min(calculated_bins, MAX_BINS))

    if combined_series.nunique() < bins:
        bins = combined_series.nunique()
        if bins < 2:
            return go.Figure().update_layout(title="Not enough unique data points to create buckets.")
    
    try:
        _, edges = pd.qcut(combined_series, q=bins, duplicates='drop', retbins=True)
    except ValueError:
        return go.Figure().update_layout(title="Could not create buckets from data distribution.")

    def format_value(v):
        if v >= 1_000_000: return f"${v/1_000_000:,.1f}M"
        if v >= 1_000: return f"${v/1_000:,.0f}k"
        return f"${v:,.0f}"

    labels = [f"{format_value(edges[i])} - {format_value(edges[i+1])}" for i in range(len(edges) - 1)]

    def process_df(df: pd.DataFrame):
        df_filtered = df.dropna(subset=[round_size_col, time_col])
        if df_filtered.empty:
            return pd.DataFrame({'bucket': labels, 'median_time': np.nan, 'sample_size': 0})

        df_filtered['bucket'] = pd.cut(df_filtered[round_size_col], bins=edges, labels=labels, include_lowest=True)
        
        stats = df_filtered.groupby('bucket', observed=False).agg(
            median_time=(time_col, 'median'),
            sample_size=(time_col, 'count')
        ).reset_index()

        # Ensure all buckets are present for consistent plotting
        stats = pd.DataFrame({'bucket': labels}).merge(stats, on='bucket', how='left')

        return stats

    stats1 = process_df(df1)
    stats2 = process_df(df2)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stats1['bucket'], y=stats1['median_time'], name=label1,
        customdata=stats1['sample_size'],
        hovertemplate='Bucket: %{x}<br>Median Time: %{y:.0f} days<br>Sample Size: %{customdata}<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        x=stats2['bucket'], y=stats2['median_time'], name=label2,
        customdata=stats2['sample_size'],
        hovertemplate='Bucket: %{x}<br>Median Time: %{y:.0f} days<br>Sample Size: %{customdata}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Time from {initial_stage.value} to {next_stage.value} by Round Size",
        xaxis_title=f"Size of {initial_stage.value} (USD) - Bucketed",
        yaxis_title=f"Median Time to {next_stage.value} (Days)",
        hovermode='x unified',
        barmode='group'
    )
    return fig 