import pandas as pd
import json
from collections import Counter
import re
from datetime import datetime

def get_round_type(title):
    if not isinstance(title, str):
        return 'Unknown'
    
    title_lower = title.lower()

    if 'pre-seed' in title_lower or 'pre seed' in title_lower:
        return 'Pre-Seed'
    if 'seed' in title_lower:
        return 'Seed'
    
    match = re.search(r'series\s+([a-z]+)', title_lower)
    if match:
        return f'Series {match.group(1).upper()}'
        
    round_types = [
        'angel', 'convertible note', 'corporate round', 'debt financing',
        'equity crowdfunding', 'grant', 'non-equity assistance',
        'private equity', 'product crowdfunding', 'secondary market',
        'undisclosed', 'venture'
    ]
    for r_type in round_types:
        if r_type in title_lower:
            return r_type.replace('-', ' ').title()

    return 'Other'

def consolidate_rounds(rounds, time_limit_days=None):
    """
    Consolidates consecutive rounds of the same type (Pre-Seed or Seed).
    If time_limit_days is specified, only consolidates if the rounds are within that limit.
    """
    processed_rounds = []
    i = 0
    while i < len(rounds):
        current_round = rounds[i]
        processed_rounds.append(current_round)
        
        round_type = get_round_type(current_round.get('title'))
        
        if round_type in ['Pre-Seed', 'Seed']:
            j = i + 1
            while j < len(rounds):
                next_round = rounds[j]
                if get_round_type(next_round.get('title')) == round_type:
                    should_consolidate = False
                    if time_limit_days:
                        date_current_str = current_round.get('announced_on')
                        date_next_str = next_round.get('announced_on')
                        if date_current_str and date_next_str:
                            date_current = datetime.strptime(date_current_str, '%Y-%m-%d')
                            date_next = datetime.strptime(date_next_str, '%Y-%m-%d')
                            if (date_next - date_current).days <= time_limit_days:
                                should_consolidate = True
                    else: # No time limit
                        should_consolidate = True
                        
                    if should_consolidate:
                        processed_rounds[-1] = next_round
                        current_round = next_round
                        j += 1
                        continue
                break
            i = j
        else:
            i += 1
            
    return processed_rounds

def consolidate_early_stage_rounds(rounds):
    """
    Consolidates all consecutive Pre-Seed and Seed rounds into a single early-stage event.
    """
    processed_rounds = []
    i = 0
    while i < len(rounds):
        current_round = rounds[i]
        
        round_type = get_round_type(current_round.get('title'))
        
        if round_type in ['Pre-Seed', 'Seed']:
            j = i + 1
            last_early_stage_round = current_round
            while j < len(rounds):
                next_round = rounds[j]
                if get_round_type(next_round.get('title')) in ['Pre-Seed', 'Seed']:
                    last_early_stage_round = next_round
                    j += 1
                else:
                    break
            processed_rounds.append(last_early_stage_round)
            i = j
        else:
            processed_rounds.append(current_round)
            i += 1
            
    return processed_rounds


def analyze_funding_rounds(df):
    """
    Analyzes the funding rounds of companies to find out what rounds follow
    pre-seed and seed rounds.
    """
    
    # For consolidated analysis with a 6-month time limit
    post_pre_seed_rounds_6m = []
    post_seed_rounds_6m = []

    # For consolidated analysis with no time limit
    post_pre_seed_rounds_nolimit = []
    post_seed_rounds_nolimit = []

    # For consolidated early-stage analysis (Pre-Seed + Seed)
    post_early_stage_rounds = []

    for _, row in df.iterrows():
        rounds_str = row['funding_rounds_list_enriched']
        
        if not rounds_str or not isinstance(rounds_str, str) or not rounds_str.strip().startswith('['):
            continue

        try:
            rounds = json.loads(rounds_str)
            # Sort rounds by date
            rounds.sort(key=lambda x: x.get('announced_on', ''))
        except (json.JSONDecodeError, TypeError):
            continue

        # --- Analysis with 6-month consolidation ---
        rounds_6m = consolidate_rounds(rounds, time_limit_days=180)
        # Find first pre-seed round and what followed
        try:
            first_pre_seed_idx = next(i for i, r in enumerate(rounds_6m) if 'pre-seed' in r.get('title', '').lower() or 'pre seed' in r.get('title', '').lower())
            if first_pre_seed_idx + 1 < len(rounds_6m):
                next_round = rounds_6m[first_pre_seed_idx + 1]
                post_pre_seed_rounds_6m.append(get_round_type(next_round.get('title')))
            else:
                post_pre_seed_rounds_6m.append('Nothing')
        except StopIteration:
            pass # No pre-seed round
        # Find first seed round and what followed
        try:
            first_seed_idx = next(i for i, r in enumerate(rounds_6m) if 'seed' in r.get('title', '').lower() and 'pre-seed' not in r.get('title', '').lower() and 'pre seed' not in r.get('title', '').lower())
            if first_seed_idx + 1 < len(rounds_6m):
                next_round = rounds_6m[first_seed_idx + 1]
                post_seed_rounds_6m.append(get_round_type(next_round.get('title')))
            else:
                post_seed_rounds_6m.append('Nothing')
        except StopIteration:
            pass # No seed round

        # --- Analysis with no time limit consolidation ---
        rounds_nolimit = consolidate_rounds(rounds)
        # Find first pre-seed round and what followed
        try:
            first_pre_seed_idx = next(i for i, r in enumerate(rounds_nolimit) if 'pre-seed' in r.get('title', '').lower() or 'pre seed' in r.get('title', '').lower())
            if first_pre_seed_idx + 1 < len(rounds_nolimit):
                next_round = rounds_nolimit[first_pre_seed_idx + 1]
                post_pre_seed_rounds_nolimit.append(get_round_type(next_round.get('title')))
            else:
                post_pre_seed_rounds_nolimit.append('Nothing')
        except StopIteration:
            pass # No pre-seed round
        # Find first seed round and what followed
        try:
            first_seed_idx = next(i for i, r in enumerate(rounds_nolimit) if 'seed' in r.get('title', '').lower() and 'pre-seed' not in r.get('title', '').lower() and 'pre seed' not in r.get('title', '').lower())
            if first_seed_idx + 1 < len(rounds_nolimit):
                next_round = rounds_nolimit[first_seed_idx + 1]
                post_seed_rounds_nolimit.append(get_round_type(next_round.get('title')))
            else:
                post_seed_rounds_nolimit.append('Nothing')
        except StopIteration:
            pass # No seed round

        # --- Analysis with consolidated early stage (Pre-Seed + Seed) ---
        rounds_early_stage = consolidate_early_stage_rounds(rounds)
        try:
            first_early_stage_idx = next(i for i, r in enumerate(rounds_early_stage) if get_round_type(r.get('title')) in ['Pre-Seed', 'Seed'])
            if first_early_stage_idx + 1 < len(rounds_early_stage):
                next_round = rounds_early_stage[first_early_stage_idx + 1]
                post_early_stage_rounds.append(get_round_type(next_round.get('title')))
            else:
                post_early_stage_rounds.append('Nothing')
        except StopIteration:
            pass # No early stage round


    return (post_pre_seed_rounds_6m, post_seed_rounds_6m), \
           (post_pre_seed_rounds_nolimit, post_seed_rounds_nolimit), \
           post_early_stage_rounds

def print_distribution_table(round_name, list_6m, list_nolimit):
    """Calculates and prints the distribution of funding rounds in a table."""
    
    # 6-month limit data
    total_6m = len(list_6m)
    counts_6m = Counter(list_6m)
    
    # No limit data
    total_nolimit = len(list_nolimit)
    counts_nolimit = Counter(list_nolimit)
    
    # Combine all round types for the table index
    all_round_types = sorted(list(set(counts_6m.keys()) | set(counts_nolimit.keys())))
    
    data = []
    for round_type in all_round_types:
        # 6-month stats
        count_6m = counts_6m.get(round_type, 0)
        perc_6m = (count_6m / total_6m) * 100 if total_6m > 0 else 0
        
        # No limit stats
        count_nolimit = counts_nolimit.get(round_type, 0)
        perc_nolimit = (count_nolimit / total_nolimit) * 100 if total_nolimit > 0 else 0
        
        data.append([round_type, count_6m, f"{perc_6m:.1f}%", count_nolimit, f"{perc_nolimit:.1f}%"])

    columns = [
        'Next Round', 
        'Count (6-Month Limit)', 'Perc. (6-Month Limit)',
        'Count (No Time Limit)', 'Perc. (No Time Limit)'
    ]
    dist_df = pd.DataFrame(data, columns=columns)
    
    # Sort by 6-month count descending
    dist_df = dist_df.sort_values(by='Count (6-Month Limit)', ascending=False).reset_index(drop=True)

    print(f"\n--- Distribution of rounds after {round_name} ---")
    print(dist_df.to_string(index=False))
    print(f"Total companies with {round_name} round (6-Month Limit): {total_6m}")
    print(f"Total companies with {round_name} round (No Time Limit): {total_nolimit}")


def print_early_stage_distribution_table(rounds_list):
    """Prints the distribution of funding rounds after a consolidated early stage."""
    total = len(rounds_list)
    if total == 0:
        print("\nNo companies found with an early-stage (Pre-Seed or Seed) round.")
        return
        
    counts = Counter(rounds_list)
    
    data = []
    for round_type, count in counts.most_common():
        percentage = (count / total) * 100
        data.append([round_type, count, f"{percentage:.1f}%"])
    
    dist_df = pd.DataFrame(data, columns=['Next Round', 'Count', 'Percentage'])
    
    print("\n--- Distribution of rounds after Early Stage (Pre-Seed + Seed Consolidated) ---")
    print(dist_df.to_string(index=False))
    print(f"Total companies with an early-stage round: {total}")


if __name__ == "__main__":
    try:
        df = pd.read_parquet('crunchbase_ready.parquet')
    except FileNotFoundError:
        print("Error: crunchbase_ready.parquet not found. Make sure it's in the same directory.")
        exit()

    (post_pre_seed_6m, post_seed_6m), \
    (post_pre_seed_nolimit, post_seed_nolimit), \
    post_early_stage = analyze_funding_rounds(df)
    
    print("Funding Round Progression Analysis:")
    print_distribution_table("Pre-seed", post_pre_seed_6m, post_pre_seed_nolimit)
    print_distribution_table("Seed", post_seed_6m, post_seed_nolimit)
    print_early_stage_distribution_table(post_early_stage)
