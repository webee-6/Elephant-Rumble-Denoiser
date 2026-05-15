"""
Batch processing script for multiple elephant calls.
"""

import os
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import warnings

from config.config import CONFIG
from src.pipeline import process_single_call

warnings.filterwarnings('ignore')


def load_and_validate_data(csv_path: str, audio_folder: str) -> pd.DataFrame:
    """
    Load CSV and validate audio file availability.
    
    Args:
        csv_path: Path to CSV file with columns [Selection, Sound_file, Start_time, End_time]
        audio_folder: Directory containing audio files
    
    Returns:
        DataFrame with processable calls (with full file paths)
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f'✅ Loaded CSV: {len(df)} calls found')
    print(f'Columns: {list(df.columns)}')
    
    # Check audio folder
    if not os.path.exists(audio_folder):
        raise FileNotFoundError(f'Audio folder not found: {audio_folder}')
    
    uploaded_filenames = set(os.listdir(audio_folder))
    print(f'✅ Found {len(uploaded_filenames)} files in: {audio_folder}')
    
    # Validate file availability
    required_filenames = set(df['Sound_file'].unique())
    available = required_filenames & uploaded_filenames
    missing = required_filenames - uploaded_filenames
    
    print(f'\n📊 File Availability:')
    print(f'   Available: {len(available)}/{len(required_filenames)}')
    
    if missing:
        print(f'   ⚠️  Missing files:')
        for f in sorted(list(missing))[:10]:
            print(f'      - {f}')
        if len(missing) > 10:
            print(f'      ... and {len(missing) - 10} more.')
    
    # Filter to processable calls
    df_processable = df[df['Sound_file'].isin(available)].copy()
    df_processable['file_path'] = df_processable['Sound_file'].apply(
        lambda x: os.path.join(audio_folder, x)
    )
    
    print(f'\n✅ Will process {len(df_processable)} calls from {len(available)} files')
    
    return df_processable


def batch_process(df: pd.DataFrame, output_dir: str = 'outputs') -> pd.DataFrame:
    """
    Process all calls in dataframe.
    
    Args:
        df: DataFrame with columns [Selection, file_path, Start_time, End_time]
        output_dir: Output directory
    
    Returns:
        DataFrame with processing results
    """
    print(f'🚀 Starting batch processing of {len(df)} calls...')
    print('='*60)
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), 
                         total=len(df),
                         desc='Processing calls'):
        
        result = process_single_call(
            audio_path=row['file_path'],
            start_time=row['Start_time'],
            end_time=row['End_time'],
            selection_id=row['Selection'],
            output_dir=output_dir
        )
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Summary
    print('\n' + '='*60)
    print('BATCH PROCESSING COMPLETE')
    print('='*60)
    print(f'Total processed: {len(results_df)}')
    print(f'Successful: {len(results_df[results_df["status"]=="success"])}')
    print(f'Failed: {len(results_df[results_df["status"]=="failed"])}')
    
    if results_df['status'].eq('failed').any():
        print('\n❌ Failed selections:')
        failed = results_df[results_df['status']=='failed'][['selection_id', 'filename', 'error']]
        print(failed.to_string())
    
    # Save results
    results_csv_path = f'{output_dir}/logs/processing_results.csv'
    results_df.to_csv(results_csv_path, index=False)
    print(f'\n📊 Results saved to: {results_csv_path}')
    
    # Summary by noise type
    if results_df['status'].eq('success').any():
        print('\n📈 Summary by noise type:')
        summary = results_df[results_df['status']=='success'].groupby('noise_type').size()
        print(summary)
    
    return results_df


def print_analysis(results_df: pd.DataFrame):
    """Print detailed analysis of processing results."""
    
    print('\n📊 RESULTS ANALYSIS')
    print('='*60)
    
    # Overall stats
    total = len(results_df)
    success = len(results_df[results_df['status']=='success'])
    failed = total - success
    
    print(f'Total Calls: {total}')
    print(f'Successful: {success} ({success/total*100:.1f}%)')
    print(f'Failed: {failed} ({failed/total*100:.1f}%)')
    
    # By noise type
    if success > 0:
        print('\nBy Noise Type:')
        noise_summary = results_df[results_df['status']=='success'].groupby('noise_type').agg({
            'selection_id': 'count',
            'duration': 'mean'
        }).rename(columns={'selection_id': 'count', 'duration': 'avg_duration_sec'})
        print(noise_summary)
        
        # Duration distribution
        successful = results_df[results_df['status']=='success']
        print(f'\nDuration Statistics:')
        print(f'  Mean: {successful["duration"].mean():.2f}s')
        print(f'  Median: {successful["duration"].median():.2f}s')
        print(f'  Min: {successful["duration"].min():.2f}s')
        print(f'  Max: {successful["duration"].max():.2f}s')
        
        # Noise source distribution
        print('\nNoise Profile Sources:')
        noise_sources = successful['noise_source'].value_counts()
        for source, count in noise_sources.items():
            print(f'  {source}: {count} ({count/len(successful)*100:.1f}%)')
