"""
Elephant Rumble Denoiser

A multi-stage DSP pipeline for removing mechanical noise from elephant bioacoustic recordings.
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

from config.config import CONFIG, PipelineConfig
from src.pipeline import process_single_call
from src.batch_process import batch_process, load_and_validate_data

__all__ = [
    'CONFIG',
    'PipelineConfig',
    'process_single_call',
    'batch_process',
    'load_and_validate_data'
]
