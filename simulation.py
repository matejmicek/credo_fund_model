import streamlit as st
import re
import json 
import numpy as np
import math
import numpy_financial as npf
import pandas as pd


# Allow larger dataframes to be styled
pd.set_option("styler.render.max_elements", 1_000_000) # Set a large number instead of None



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

            # --- NEW: Dynamically create investments with randomized ticket sizes ---
            invested_capital_in_bucket = 0
            attempts = 0
            while invested_capital_in_bucket < bucket_capital:
                # Randomize ticket size: +-15%
                ticket_variance = np.random.uniform(-0.15, 0.15)
                randomized_ticket = avg_ticket * (1 + ticket_variance)
                
                # Round to nearest 50k
                ticket_size = round(randomized_ticket * 20) / 20

                # Ensure we don't overallocate the bucket
                if invested_capital_in_bucket + ticket_size > bucket_capital:
                    if attempts > 3:
                        break
                    attempts += 1
                    continue

                invested_capital_in_bucket += ticket_size

                # Determine investment timing
                deploy_pcts = np.array([
                    bucket.get('deploy_y1', 0), bucket.get('deploy_y2', 0),
                    bucket.get('deploy_y3', 0), bucket.get('deploy_y4', 0)
                ])
                if deploy_pcts.sum() == 0: continue
                deploy_probs = deploy_pcts / deploy_pcts.sum()

                invest_year = np.random.choice([0, 1, 2, 3], p=deploy_probs)
                month_offset = np.random.randint(0, 12)
                invest_month = invest_year * 12 + month_offset

                all_investments.append({
                    'bucket_key': i_str,
                    'bucket': bucket,
                    'invest_month': invest_month,
                    'invest_year': invest_year,
                    'ticket_size': ticket_size  # Store the actual ticket size
                })

        # Process each investment through its lifecycle
        for investment_idx, investment in enumerate(all_investments):
            bucket_key = investment['bucket_key']
            bucket = investment['bucket']
            invest_month = investment['invest_month']
            investment_year = investment['invest_year']
            ticket_size = investment['ticket_size'] # Use actual ticket size

            # Sample entry valuation for this specific investment
            entry_valuation_min = bucket.get('entry_valuation_min', 0)
            entry_valuation_max = bucket.get('entry_valuation_max', 0)
            entry_valuation = np.random.uniform(entry_valuation_min, entry_valuation_max) if entry_valuation_max > entry_valuation_min else entry_valuation_min

            # Track initial investment stats
            initial_capital_invested_by_bucket[bucket_key] += ticket_size
            initial_investment_count_by_bucket[bucket_key] += 1

            # Initial investment cash flow
            cash_flows[invest_month] -= ticket_size
            total_invested_cash += ticket_size

            # --- Ownership and Return Calculation ---
            initial_ownership_pct = (ticket_size / entry_valuation * 100) if entry_valuation > 0 else 0

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
                follow_on_amount = ticket_size * (follow_on_size_pct / 100.0)

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
                        if follow_on_amount > 0:
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
                'initial_check': ticket_size,
                'initial_ownership': initial_ownership_pct,
                'entry_valuation': entry_valuation,
                'follow_on': "Yes" if follow_on_investment > 0 else "No",
                'follow_on_check': follow_on_investment,
                'ownership_after_follow_on': ownership_before_exit_pct,
                'follow_on_dilution_pct': (float(chosen_strategy.get('dilution_pct')) if (chosen_strategy is not None) else bucket.get('follow_on_dilution_pct', 15)),
                'ownership_after_follow_on_dilution': ((initial_ownership_pct * (1 - float(chosen_strategy.get('dilution_pct', 15)) / 100.0)) if (chosen_strategy is not None) else initial_ownership_pct),
                'ownership_from_follow_on': ((follow_on_investment / (entry_valuation * float(chosen_strategy.get('valuation_multiple', 2.0))) * 100.0) if (follow_on_investment > 0 and chosen_strategy is not None and entry_valuation * float(chosen_strategy.get('valuation_multiple', 2.0)) > 0) else 0.0),
                'final_ownership_at_exit': final_ownership_pct,
                'status': status,
                'exit_year': ((exit_month // 12) + 1) if status != "Active" else None,
                'exit_valuation': exit_valuation if status != "Active" else None,
                'net_return': realized_value,
                'exit_scenario': chosen_scenario.get('name', 'N/A'),
                'exit_dilution_pct': (scenario_exit_dilution_pct if status != "Active" else None)
            })

        # --- NEW: Deploy leftover capital ---
        total_initial_invested = sum(initial_capital_invested_by_bucket.values())
        total_follow_on_spent = sum(follow_on_capital_spent_by_bucket.values())
        leftover_capital = investable_capital - (total_initial_invested + total_follow_on_spent)

        # Get bucket probabilities for weighted random choice
        bucket_keys = list(fund_model['buckets'].keys())
        bucket_probs = np.array([b['percentage'] for b in fund_model['buckets'].values()], dtype=float)
        bucket_probs /= bucket_probs.sum()

        while leftover_capital > 0.05: # Minimum threshold to attempt investment
            # Probabilistically choose a bucket for the new investment
            chosen_bucket_key = np.random.choice(bucket_keys, p=bucket_probs)
            bucket = fund_model['buckets'][chosen_bucket_key]
            avg_ticket = bucket.get('avg_ticket', 0)
            if avg_ticket <= 0 or avg_ticket > leftover_capital:
                break # Stop if no suitable investment can be made

            # Create a new primary investment
            ticket_variance = np.random.uniform(-0.15, 0.15)
            randomized_ticket = avg_ticket * (1 + ticket_variance)
            ticket_size = min(round(randomized_ticket * 20) / 20, leftover_capital)

            if ticket_size < 0.01:
                break

            # Investment timing (e.g., place it in year 4)
            invest_month = np.random.randint(0, 36)

            # Process this new investment's lifecycle
            # (This is a simplified version of the main loop's logic)
            entry_valuation = np.random.uniform(bucket.get('entry_valuation_min', 0), bucket.get('entry_valuation_max', 0))
            initial_ownership_pct = (ticket_size / entry_valuation * 100) if entry_valuation > 0 else 0
            
            # Update tracking vars
            initial_capital_invested_by_bucket[chosen_bucket_key] += ticket_size
            initial_investment_count_by_bucket[chosen_bucket_key] += 1
            cash_flows[invest_month] -= ticket_size
            total_invested_cash += ticket_size
            leftover_capital -= ticket_size
            
            # Simplified outcome - no follow-on for these leftover investments for now
            # You could expand this to include follow-ons if desired
            scenarios = bucket.get('scenarios', [])
            realized_value = 0
            exit_valuation = 0
            final_ownership_pct = 0
            status = "Active"
            exit_month = -1
            chosen_scenario_name = "N/A"
            scenario_exit_dilution_pct = 0

            if scenarios:
                probs = np.array([s.get('probability', 0) for s in scenarios], dtype=float)
                if probs.sum() > 0:
                    probs /= probs.sum()
                    chosen_scenario = scenarios[np.random.choice(len(scenarios), p=probs)]
                    chosen_scenario_name = chosen_scenario.get('name', 'N/A')
                    scenario_exit_dilution_pct = chosen_scenario.get('exit_dilution_pct', 20)
                    
                    exit_valuation = np.random.uniform(chosen_scenario.get('exit_valuation_min', 0), chosen_scenario.get('exit_valuation_max', 0))
                    time_to_exit_months = np.random.randint(chosen_scenario.get('exit_year_min', 5) * 12, chosen_scenario.get('exit_year_max', 8) * 12 + 1)
                    exit_month = invest_month + time_to_exit_months

                    if exit_month < FUND_LIFE_MONTHS:
                        final_ownership_pct = initial_ownership_pct * (1 - scenario_exit_dilution_pct / 100)
                        realized_value = (final_ownership_pct / 100) * exit_valuation
                        realized_value_by_bucket[chosen_bucket_key] += realized_value
                        cash_flows[exit_month] += realized_value
                        total_realized_value += realized_value
                        status = "Exited" if exit_valuation > 0 else "Failed"

            # Append to portfolio details
            portfolio_details.append({
                'company_id': f"Company {len(portfolio_details) + 1}",
                'investment_year': (invest_month // 12) + 1,
                'stage': bucket.get('name', 'N/A'),
                'initial_check': ticket_size,
                'initial_ownership': initial_ownership_pct,
                'entry_valuation': entry_valuation,
                'follow_on': "No",
                'follow_on_check': 0.0,
                'ownership_after_follow_on': initial_ownership_pct,
                'follow_on_dilution_pct': 0,
                'ownership_after_follow_on_dilution': initial_ownership_pct,
                'ownership_from_follow_on': 0.0,
                'final_ownership_at_exit': final_ownership_pct,
                'status': status,
                'exit_year': ((exit_month // 12) + 1) if status != "Active" else None,
                'exit_valuation': exit_valuation if status != "Active" else None,
                'net_return': realized_value,
                'exit_scenario': chosen_scenario_name,
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