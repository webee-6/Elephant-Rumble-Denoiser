#!/usr/bin/env python3
"""
KUMKI RADAR - Elephant Rumble Denoiser

Usage:
    python main.py --csv /path/to/calls.csv --audio /path/to/audio/folder
    python main.py --csv calls.csv --audio ./audio --test  # Test on first call only
"""

import os
import sys
import argparse
import warnings
import shutil
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))
from config.config import CONFIG
from src.batch_process import load_and_validate_data, batch_process, print_analysis
from src.pipeline import process_single_call
warnings.filterwarnings('ignore')

def setup_directories():
    os.makedirs('outputs/audio', exist_ok=True)
    os.makedirs('outputs/spectrograms', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)

def test_single_call(df):
    test_row = df.iloc[0]
    print(' Testing pipeline on single call...')
    print(f'   File: {test_row["Sound_file"]}')
    print(f'   Time: {test_row["Start_time"]:.2f}s - {test_row["End_time"]:.2f}s')
    print(f'   Selection: {test_row["Selection"]}')
    test_result = process_single_call(
        audio_path=test_row['file_path'],
        start_time=test_row['Start_time'],
        end_time=test_row['End_time'],
        selection_id=test_row['Selection']
    )
    print('\n' + '='*60)
    print('TEST RESULT:')
    print('='*60)
    for key, value in test_result.items():
        if key != 'noise_validation':
            print(f'{key:20s}: {value}')
    
    if test_result['status'] == 'success':
        print('\n Test successful! Pipeline is working.')
        print(f'\n Outputs:')
        print(f'   Audio: {test_result["output_audio"]}')
        print(f'   Spectrogram: {test_result["spectrogram"]}')
        print(f'   Comparison: {test_result["comparison_plot"]}')
    else:
        print(f'\n Test failed: {test_result["error"]}')
    return test_result

def create_archive():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_name = f'elephant_denoiser_results_{timestamp}'
    
    print(f'\n Creating archive: {zip_name}.zip')
    shutil.make_archive(zip_name, 'zip', 'outputs')
    
    full_zip_path = os.path.abspath(f"{zip_name}.zip")
    print(f' Archive created successfully!')
    print(f' Location: {full_zip_path}')
    print(f' Size: {os.path.getsize(full_zip_path) / (1024*1024):.1f} MB')
    
    return full_zip_path


def main():
    parser = argparse.ArgumentParser(description='Elephant Rumble Denoiser - Remove mechanical noise from bioacoustic recordings')
    parser.add_argument('--csv', required=True, help='Path to CSV file with call annotations')
    parser.add_argument('--audio', required=True, help='Path to folder containing audio files')
    parser.add_argument('--test', action='store_true', help='Test on first call only (don\'t process full batch)')
    parser.add_argument('--no-archive', action='store_true', help='Skip creating zip archive at the end')
    parser.add_argument('--config', help='Path to custom config file (optional)')
    args = parser.parse_args()
    
    # Banner
    print('='*60)
    print('KUMKI RADAR - ELEPHANT RUMBLE DENOISER')
    print('='*60)
    
    # Setup
    setup_directories()
    CONFIG.print_summary()
    
    # Load and validate data
    print('\n Loading data...')
    df = load_and_validate_data(args.csv, args.audio)
    
    if len(df) == 0:
        print(' No processable calls found. Exiting.')
        return 1
    
    # Test mode
    if args.test:
        test_single_call(df)
        return 0
    
    # Batch process
    results_df = batch_process(df, output_dir='outputs')
    
    # Analysis
    print_analysis(results_df)
    
    # Create archive
    if not args.no_archive:
        create_archive()
    
    print('\n All done!')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
