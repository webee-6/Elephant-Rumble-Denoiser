"""
Setup script for Elephant Rumble Denoiser
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="elephant-rumble-denoiser",
    version="1.0.0",
    author="Balakrishnan Arumugam",
    author_email="your.email@example.com",  # Update this
    description="Multi-stage DSP pipeline for removing mechanical noise from elephant bioacoustic recordings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Krish-008/elephant-rumble-denoiser",
    project_urls={
        "Bug Tracker": "https://github.com/Krish-008/elephant-rumble-denoiser/issues",
        "Documentation": "https://github.com/Krish-008/elephant-rumble-denoiser#readme",
        "Source Code": "https://github.com/Krish-008/elephant-rumble-denoiser",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "bioacoustics",
        "signal-processing",
        "noise-reduction",
        "elephant",
        "audio-processing",
        "dsp",
        "spectral-gating",
        "hpss",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "noisereduce>=3.0.0",
        "scipy>=1.9.0",
        "pandas>=1.5.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "elephant-denoiser=main:main",
        ],
    },
)
