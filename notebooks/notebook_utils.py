"""
Helper utilities for running the pipeline from Jupyter notebooks.
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
notebook_dir = Path.cwd()
project_root = notebook_dir.parent if 'notebooks' in str(notebook_dir) else notebook_dir
sys.path.insert(0, str(project_root))


def setup_notebook_environment():
    """
    Setup environment for notebook execution.
    Creates output directories and configures paths.
    """
    # Create output directories relative to project root
    output_dir = project_root / 'outputs'
    os.makedirs(output_dir / 'audio', exist_ok=True)
    os.makedirs(output_dir / 'spectrograms', exist_ok=True)
    os.makedirs(output_dir / 'logs', exist_ok=True)
    
    print('Notebook environment ready')
    print(f'Project root: {project_root}')
    print(f'Outputs: {output_dir}')
    
    return str(output_dir)


def display_result(result, show_audio=True):
    """
    Display processing result with nice formatting.
    
    Args:
        result: Result dictionary from process_single_call
        show_audio: Whether to display audio player (requires IPython)
    """
    from IPython.display import Image, Audio, display
    
    print('\n' + '='*60)
    print('PROCESSING RESULT')
    print('='*60)
    
    for key, value in result.items():
        if key != 'noise_validation':
            print(f'{key:20s}: {value}')
    
    if result['status'] == 'success':
        print('\n📊 Visualizations:')
        
        # Display comparison spectrogram
        if 'comparison_plot' in result and os.path.exists(result['comparison_plot']):
            print('\nBefore/After Comparison:')
            display(Image(result['comparison_plot']))
        
        # Display cleaned spectrogram
        if 'spectrogram' in result and os.path.exists(result['spectrogram']):
            print('\nCleaned Spectrogram:')
            display(Image(result['spectrogram']))
        
        # Play cleaned audio
        if show_audio and 'output_audio' in result and os.path.exists(result['output_audio']):
            print('\n🔊 Cleaned Audio:')
            display(Audio(result['output_audio']))
    else:
        print(f'\nProcessing failed: {result["error"]}')


def quick_test(df, selection_index=0, show_audio=True):
    """
    Quick test on a single call from dataframe.
    
    Args:
        df: DataFrame with call data
        selection_index: Index to test (default: 0 = first row)
        show_audio: Whether to play audio
    
    Returns:
        Result dictionary
    """
    from src.pipeline import process_single_call
    
    test_row = df.iloc[selection_index]
    
    print(f'Testing on selection {selection_index}')
    print(f'   File: {test_row["Sound_file"]}')
    print(f'   Time: {test_row["Start_time"]:.2f}s - {test_row["End_time"]:.2f}s')
    
    result = process_single_call(
        audio_path=test_row['file_path'],
        start_time=test_row['Start_time'],
        end_time=test_row['End_time'],
        selection_id=test_row['Selection']
    )
    
    display_result(result, show_audio=show_audio)
    
    return result
