import pandas as pd
import json
from typing import Dict, Any, List

# Project-specific imports
from data_utils import FundingStage
from constants import TIER1_INVESTORS

# --- Constants ---
INPUT_FILE = 'crunchbase.csv'
OUTPUT_FILE = 'crunchbase_ready.parquet'
MANUAL_FILE = 'dataset_manual.txt'

# --- Manual Generation ---

def initialize_manual():
    """Creates or clears the dataset manual file."""
    with open(MANUAL_FILE, 'w') as f:
        f.write("--- Crunchbase Dataset Manual ---\n")
        f.write("This file provides a detailed description of the features in the processed dataset.\n\n")
    print(f"Initialized manual file: {MANUAL_FILE}")

def append_to_manual(df: pd.DataFrame, column_name: str, description: str):
    """Appends details about a new feature to the dataset manual."""
    with open(MANUAL_FILE, 'a') as f:
        f.write(f"--- Feature: {column_name} ---\n")
        f.write(f"Description: {description}\n")
        
        col = df[column_name]
        
        if pd.api.types.is_numeric_dtype(col.dropna()):
            stats = col.describe()
            f.write("Type: Numeric\n")
            f.write(f"Stats:\n{stats.to_string()}\n\n")
        else:
            f.write("Type: Object/Categorical\n")
            f.write(f"Number of Unique Values: {col.nunique()}\n")
            # Show top 5 value counts for non-numeric columns, if there are any values
            if not col.value_counts().empty:
                f.write(f"Top 5 Value Counts:\n{col.value_counts().head().to_string()}\n\n")
            else:
                f.write("No values to display.\n\n")
    print(f"Appended details for '{column_name}' to manual.")

# --- Data Processing Functions ---

def build_funding_round_mapping(df: pd.DataFrame) -> Dict[str, Any]:
    """Creates a mapping from funding round ID to its details."""
    print("Building funding round mapping...")
    mapping = {}
    for investors_data in df['investors'].dropna():
        try:
            investors_list = json.loads(investors_data)
            for investment in investors_list:
                if isinstance(investment, dict) and 'funding_round' in investment:
                    funding_round = investment['funding_round']
                    if isinstance(funding_round, dict) and 'id' in funding_round:
                        mapping[funding_round['id']] = funding_round
        except (json.JSONDecodeError, TypeError):
            continue
    print(f"Mapped {len(mapping):,} unique funding rounds.")
    return mapping

def enrich_funding_rounds(df: pd.DataFrame, mapping: Dict[str, Any]) -> pd.DataFrame:
    """Enriches the funding rounds list with details from the mapping."""
    print("Enriching funding rounds...")
    enriched_data = []
    for _, row in df.iterrows():
        if pd.isna(row['funding_rounds_list']):
            enriched_data.append(None)
            continue
        try:
            rounds = json.loads(row['funding_rounds_list'])
            enriched_rounds = [
                {**r, **mapping.get(r.get('id', ''), {})}
                for r in rounds if isinstance(r, dict)
            ]
            enriched_data.append(json.dumps(enriched_rounds))
        except (json.JSONDecodeError, TypeError):
            enriched_data.append(None)
    df['funding_rounds_list_enriched'] = enriched_data
    print("Funding rounds enriched.")
    return df

def add_founding_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'founding_year' column by parsing 'founded_date'."""
    print("Adding 'founding_year' column...")
    df['founding_year'] = pd.to_datetime(df['founded_date'], errors='coerce').dt.year
    print("'founding_year' column added.")
    return df

def add_total_funding_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates total funding by parsing the 'funding_rounds' column.
    Extracts the 'value_usd' from the JSON content.
    """
    print("Adding 'total_funding_usd' column from 'funding_rounds'...")

    def parse_funding_from_rounds(funding_rounds_json: Any) -> float:
        """
        Parses the 'funding_rounds' JSON string to extract the total funding in USD.
        """
        if pd.isna(funding_rounds_json) or not isinstance(funding_rounds_json, str):
            return 0.0
        
        try:
            funding_rounds_data = json.loads(funding_rounds_json)
            if isinstance(funding_rounds_data, dict):
                value_dict = funding_rounds_data.get('value', {})
                if isinstance(value_dict, dict):
                    value_usd = value_dict.get('value_usd')
                    if value_usd is not None:
                        return float(value_usd)
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0.0
        
        return 0.0

    if 'funding_rounds' in df.columns:
        df['total_funding_usd'] = df['funding_rounds'].apply(parse_funding_from_rounds)
        print("'total_funding_usd' column added from 'funding_rounds'.")
    else:
        df['total_funding_usd'] = 0.0
        print("Warning: 'funding_rounds' column not found. 'total_funding_usd' initialized to 0.")
    
    return df

def add_tier1_investor_flags(df: pd.DataFrame, tier1_investors: List[str]) -> pd.DataFrame:
    """Adds boolean flags indicating if a company raised a specific stage from a Tier-1 investor."""
    print("Adding Tier-1 investor flags...")
    
    # Initialize a column for each funding stage to track if a Tier-1 investor participated.
    for stage in FundingStage:
        df[f'raised_{stage.name.lower()}_from_tier1'] = False
        df[f'raised_{stage.name.lower()}'] = False

    investor_list_set = set(tier1_investors)

    for index, row in df.iterrows():
        if pd.isna(row['investors']):
            continue

        try:
            investors_data = json.loads(row['investors'])
            
            # This dictionary will track which stages have been confirmed for this company.
            stages_funded_by_tier1 = {}
            stages_raised = {}

            for investment in investors_data:
                if not isinstance(investment, dict):
                    continue
                
                funding_round = investment.get('funding_round', {})
                investor_info = investment.get('investor', {})

                if not isinstance(funding_round, dict) or not isinstance(investor_info, dict):
                    continue
                
                round_name = funding_round.get('value')
                investor_name = investor_info.get('value')

                if not round_name or not investor_name:
                    continue

                # Check all defined funding stages
                for stage in FundingStage:
                    if round_name.strip().startswith(stage.value):
                        stages_raised[stage] = True
                        # If the investor for this round is in our Tier-1 list...
                        if investor_name in investor_list_set:
                            # ...mark this stage as funded by a Tier-1 for this company.
                            stages_funded_by_tier1[stage] = True

            # After checking all investments for the company, update the DataFrame.
            for stage, was_funded_by_tier1 in stages_funded_by_tier1.items():
                if was_funded_by_tier1:
                    df.loc[index, f'raised_{stage.name.lower()}_from_tier1'] = True
            
            for stage, was_raised in stages_raised.items():
                if was_raised:
                    df.loc[index, f'raised_{stage.name.lower()}'] = True

        except (json.JSONDecodeError, TypeError):
            continue
            
    print("Tier-1 investor flags added.")
    return df

def add_round_size_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds columns for the size of each specific funding round."""
    print("Adding round size columns...")

    stage_columns = {stage: f"{stage.name.lower()}_round_size_usd" for stage in FundingStage}
    for stage, col_name in stage_columns.items():
        df[col_name] = pd.NA

    for index, row in df.iterrows():
        try:
            rounds = json.loads(row['funding_rounds_list_enriched'])
            for r in rounds:
                if not isinstance(r, dict):
                    continue

                round_name = r.get('value', '')
                # Correctly access the nested 'money_raised' -> 'value_usd' field
                money_raised = r.get('money_raised', {}).get('value_usd') if isinstance(r.get('money_raised'), dict) else None

                if not round_name or pd.isna(money_raised):
                    continue

                for stage, col_name in stage_columns.items():
                    if round_name.strip().startswith(stage.value):
                        # Take the first one found for that stage type
                        if pd.isna(df.loc[index, col_name]):
                             df.loc[index, col_name] = money_raised
                        break # Move to next round in list
        except (json.JSONDecodeError, TypeError):
            continue

    # Convert columns to numeric, coercing errors
    for col_name in stage_columns.values():
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    
    print("Round size columns added.")
    return df

def add_round_timing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds columns for round dates and the time between consecutive rounds."""
    print("Adding round timing columns...")

    # 1. Extract the date for each round type
    stage_date_cols = {stage: f"{stage.name.lower()}_round_date" for stage in FundingStage}
    for col_name in stage_date_cols.values():
        df[col_name] = pd.NaT

    for index, row in df.iterrows():
        try:
            rounds = json.loads(row['funding_rounds_list_enriched'])
            for r in rounds:
                if not isinstance(r, dict): continue
                
                round_name = r.get('value', '')
                announced_on = r.get('announced_on')

                if not round_name or pd.isna(announced_on): continue

                for stage, col_name in stage_date_cols.items():
                    if round_name.strip().startswith(stage.value):
                        if pd.isna(df.loc[index, col_name]):
                            df.loc[index, col_name] = announced_on
                        break
        except (json.JSONDecodeError, TypeError):
            continue
    
    for col_name in stage_date_cols.values():
        df[col_name] = pd.to_datetime(df[col_name], errors='coerce')

    # 2. Calculate time between consecutive rounds
    all_stages = list(FundingStage)
    for i in range(len(all_stages) - 1):
        from_stage = all_stages[i]
        to_stage = all_stages[i+1]
        
        from_col = stage_date_cols[from_stage]
        to_col = stage_date_cols[to_stage]
        
        time_diff_col = f"time_from_{from_stage.name.lower()}_to_{to_stage.name.lower()}_days"
        
        # Calculate timedelta and convert to days
        df[time_diff_col] = (df[to_col] - df[from_col]).dt.days
        # Ensure time difference is positive
        df[time_diff_col] = df[time_diff_col].apply(lambda x: x if x > 0 else pd.NA)

    print("Round timing columns added.")
    return df

# --- Main Execution ---

def main():
    """Main function to orchestrate the data preparation pipeline."""
    initialize_manual()
    
    print(f"Loading raw data from '{INPUT_FILE}'...")
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False)
        print("Raw data loaded successfully.")
    except FileNotFoundError:
        print(f"Fatal Error: The data file '{INPUT_FILE}' was not found.")
        return

    # --- Step 1: Enrich Funding Rounds ---
    mapping = build_funding_round_mapping(df)
    df = enrich_funding_rounds(df, mapping)
    append_to_manual(
        df, 
        'funding_rounds_list_enriched',
        "A JSON string containing a list of funding rounds for the company, enriched with details (e.g., investment type, money raised) from the 'investors' column."
    )

    # --- Step 2: Add Founding Year ---
    df = add_founding_year_column(df)
    append_to_manual(
        df,
        'founding_year',
        "The year the company was founded, extracted from the 'founded_date' column. Allows for time-based analysis."
    )

    # --- Step 3: Add Total Funding ---
    df = add_total_funding_column(df)
    append_to_manual(
        df,
        'total_funding_usd',
        "The total funding raised by the company in USD, aggregated from all its funding rounds. A value of 0.0 indicates no funding information was available."
    )
    
    # --- Step 4: Add Tier-1 Investor Flags ---
    df = add_tier1_investor_flags(df, TIER1_INVESTORS)
    for stage in FundingStage:
        col_name = f'raised_{stage.name.lower()}_from_tier1'
        append_to_manual(
            df,
            col_name,
            f"A boolean flag that is TRUE if the company raised a {stage.value} from a Tier-1 investor."
        )
        
        col_name_raised = f'raised_{stage.name.lower()}'
        append_to_manual(
            df,
            col_name_raised,
            f"A boolean flag that is TRUE if the company raised a {stage.value}, regardless of investor."
        )

    # --- Step 5: Add Round Size Columns ---
    df = add_round_size_columns(df)
    for stage in FundingStage:
        col_name = f'{stage.name.lower()}_round_size_usd'
        append_to_manual(
            df,
            col_name,
            f"The amount in USD raised during the company's {stage.value}. Only the first recorded round of this type is used."
        )

    # --- Step 6: Add Round Timing Columns ---
    df = add_round_timing_columns(df)
    for i in range(len(list(FundingStage)) - 1):
        from_stage = list(FundingStage)[i]
        to_stage = list(FundingStage)[i+1]
        col_name = f"time_from_{from_stage.name.lower()}_to_{to_stage.name.lower()}_days"
        append_to_manual(
            df,
            col_name,
            f"The time in days between the company's {from_stage.value} and {to_stage.value}."
        )
        
    # --- Step 7: Clean and Finalize Dataset ---
    print("Cleaning final dataset...")
    required_columns = ['funding_rounds_list_enriched', 'founding_year', 'investors', 'id', 'total_funding_usd']
    # Print statistics about missing data for each required column
    print("\nMissing data statistics:")
    for col in required_columns:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        print(f"  - {col}: {missing_count:,} rows missing ({missing_pct:.1f}%)")
    final_df = df.dropna(subset=required_columns).copy()
    print(f"Final dataset has {len(final_df):,} rows after cleaning.")

    # --- Step 8: Select and Drop Unused Columns ---
    print("Selecting final columns and dropping unused ones...")
    tier1_flag_columns = [f'raised_{stage.name.lower()}_from_tier1' for stage in FundingStage]
    raised_flag_columns = [f'raised_{stage.name.lower()}' for stage in FundingStage]
    round_size_columns = [f'{stage.name.lower()}_round_size_usd' for stage in FundingStage]
    
    time_diff_columns = []
    for i in range(len(list(FundingStage)) - 1):
        from_stage = list(FundingStage)[i]
        to_stage = list(FundingStage)[i+1]
        time_diff_columns.append(f"time_from_{from_stage.name.lower()}_to_{to_stage.name.lower()}_days")

    final_columns_to_keep = [
        'id',
        'founding_year',
        'total_funding_usd',
        'funding_rounds_list_enriched'
    ] + tier1_flag_columns + raised_flag_columns + round_size_columns + time_diff_columns
    
    final_df = final_df[final_columns_to_keep]
    print(f"Kept {len(final_df.columns)} columns for the final dataset.")
    
    # --- Step 9: Save Processed Data ---
    print(f"Saving processed data to '{OUTPUT_FILE}'...")
    final_df.to_parquet(OUTPUT_FILE, index=False)
    
    print("\nData preparation complete!")
    print(f"✅ Processed file saved: {OUTPUT_FILE}")
    print(f"✅ Dataset manual saved: {MANUAL_FILE}")

if __name__ == "__main__":
    main() 