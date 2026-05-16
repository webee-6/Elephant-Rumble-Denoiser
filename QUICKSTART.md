# 🚀 Quick Start Guide

## Installation (1 minute)

```bash
git clone https://github.com/Krish-008/elephant-rumble-denoiser.git
cd elephant-rumble-denoiser
pip install -r requirements.txt
```

## Test Run (2 minutes)

```bash
# Test on single call
python main.py --csv data/example.csv --audio data/audio --test
```

## Full Processing (5 minutes)

```bash
# Process all calls in CSV
python main.py --csv data/calls.csv --audio data/audio
```

## Results

Outputs saved to:
- 📁 `outputs/audio/` - Cleaned WAV files
- 📊 `outputs/spectrograms/` - Before/After visualizations
- 📋 `outputs/logs/` - Processing results CSV

---

## Next Steps

- 📖 Read the [full README](README.md) for detailed documentation
- 🔧 See [EXAMPLES.md](EXAMPLES.md) for Python API usage
- ⚙️ Customize parameters in `config/config.py`
- 📓 Explore `notebooks/interactive_demo.ipynb` for interactive testing

## Need Help?

- 🐛 [Report Issues](https://github.com/Krish-008/elephant-rumble-denoiser/issues)
- 💡 [Request Features](https://github.com/Krish-008/elephant-rumble-denoiser/issues/new)
- 📧 Contact: your.email@example.com
