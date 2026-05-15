"""
Parameter tuning and testing utilities.
"""

import pandas as pd
from typing import List

from config.config import CONFIG
from src.pipeline import process_single_call


def test_parameters(df: pd.DataFrame,
                   selection_id: int,
                   prop_decrease_values: List[float] = [0.75, 0.85, 0.90],
                   hpss_margins: List[float] = [2.0, 3.0, 4.0]) -> pd.DataFrame:
    """
    Test multiple parameter combinations on a single call.
    
    Args:
        df: DataFrame with call data
        selection_id: Selection ID to test
        prop_decrease_values: List of prop_decrease values to test
        hpss_margins: List of HPSS margin values to test
    
    Returns:
        DataFrame with test results
    """
    # Get the call
    row = df[df['Selection'] == selection_id].iloc[0]
    
    print(f'Testing on Selection {selection_id}: {row["Sound_file"]}')
    
    test_results = []
    
    for prop_dec in prop_decrease_values:
        for margin in hpss_margins:
            # Temporarily modify config
            original_prop = CONFIG.noise_params['vehicle']['prop_decrease']
            original_margin = CONFIG.hpss_margin
            
            CONFIG.noise_params['vehicle']['prop_decrease'] = prop_dec
            CONFIG.hpss_margin = margin
            
            result = process_single_call(
                audio_path=row['file_path'],
                start_time=row['Start_time'],
                end_time=row['End_time'],
                selection_id=selection_id
            )
            
            result['prop_decrease'] = prop_dec
            result['hpss_margin'] = margin
            test_results.append(result)
            
            # Restore
            CONFIG.noise_params['vehicle']['prop_decrease'] = original_prop
            CONFIG.hpss_margin = original_margin
    
    # Display results
    test_df = pd.DataFrame(test_results)
    print('\nParameter Test Results:')
    print(test_df[['prop_decrease', 'hpss_margin', 'status', 'duration']].to_string())
    
    return test_df
