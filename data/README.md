# 📂 Data Directory

This directory contains input data for the elephant rumble denoiser.

## Structure

```
data/
├── example_annotations.csv    # Example annotated CSV (template)
├── labels.json               # ML labels (generated)
├── your_calls.csv           # Your elephant call timestamps
└── your_audio/              # Your audio files (not tracked in git)
```

---

## Creating Your Labels

### Option 1: Add Labels to Your CSV

Edit your existing CSV and add a `CallType` column:

```csv
Selection,Sound_file,Start_time,End_time,CallType
1,recording1.wav,5.2,8.7,contact
2,recording2.wav,12.1,15.3,greeting
3,recording3.wav,22.4,25.1,alarm
```

**Label types:**
- `contact` - Social bonding call
- `greeting` - Reunion call
- `alarm` - Warning/danger call
- `mating` - Reproductive call
- `unknown` - Unclear/ambiguous

Then convert to JSON:

```bash
python scripts/create_labels.py \
    --csv data/your_calls.csv \
    --output data/labels.json
```

### Option 2: Manual JSON Creation

Create `data/labels.json` directly:

```json
{
  "selection_001_recording1_cleaned.wav": 0,
  "selection_002_recording2_cleaned.wav": 1,
  "selection_003_recording3_cleaned.wav": 2
}
```

Where:
- 0 = contact
- 1 = greeting
- 2 = alarm
- 3 = mating

### Option 3: Use Annotation Tools

See [LABEL_CREATION.md](../docs/LABEL_CREATION.md) for:
- Raven Pro
- Label Studio
- Audacity
- Active learning approaches

---

## Example Workflow

```bash
# 1. Place your CSV here
cp /path/to/your/calls.csv data/my_calls.csv

# 2. Denoise all calls
python main.py --csv data/my_calls.csv --audio /path/to/audio

# 3. Listen and annotate
# Open outputs/audio/*.wav in Raven/Audacity
# Add CallType column to CSV

# 4. Create labels
python scripts/create_labels.py \
    --csv data/my_calls_annotated.csv \
    --output data/labels.json

# 5. Train ML model
python -c "
from src.ai_training import RumbleTrainer
trainer = RumbleTrainer('outputs/audio', 'data/labels.json')
results = trainer.train_random_forest()
"
```

---

## Label Quality Tips

1. **Be consistent** - Use clear definitions for each call type
2. **Multiple annotators** - Have 2-3 people label same calls
3. **Confidence scores** - Mark uncertain labels
4. **Start small** - Label 50-100 calls first, train model, then continue
5. **Review mistakes** - Look at misclassified calls, refine definitions

See [LABEL_CREATION.md](../docs/LABEL_CREATION.md) for detailed guidance.
