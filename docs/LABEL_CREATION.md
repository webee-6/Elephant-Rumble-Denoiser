# 🏷️ Label Creation Guide for Elephant Rumbles

Complete guide to creating labels for supervised machine learning.

---

## What Are Labels?

**Labels** are the "answers" you teach the AI:
- **Classification**: What type of call is this? (0=contact, 1=greeting, 2=alarm)
- **Individual ID**: Which elephant? (0=Ellie, 1=Dumbo, 2=Jumbo)
- **Detection**: Is this a rumble? (0=noise, 1=rumble)

---

## 📋 Method 1: Manual Annotation (Gold Standard)

### Using Raven Pro / Audacity

**Best for**: Research-grade labels, small datasets (<500 calls)

#### Step 1: Open in Raven Pro (or Audacity)

```
File → Open Sound Files → select your cleaned audio
```

#### Step 2: View Spectrogram

```
Configure → New Spectrogram View
Settings:
  - Window: Hann
  - Size: 4096
  - Overlap: 75%
  - Frequency: 0-500 Hz (for elephants)
```

#### Step 3: Annotate Each Call

**Listen + Look at spectrogram:**
- Contact call: Low frequency (15-35 Hz), long duration (>2s), steady
- Greeting: Higher pitch (25-50 Hz), shorter, rising contour
- Alarm: Rapid, loud, 40-100 Hz, choppy

**Create annotation:**
```
Table → New Selection
For each call:
  - Mark start/end time
  - Add label in "Class" column
  - Add notes (elephant ID, context, confidence)
```

#### Step 4: Export Annotations

```
File → Export Selection Table As → CSV
```

**Example output (Raven format):**
```csv
Selection,View,Channel,Begin Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Class,Individual,Notes
1,Spectrogram 1,1,5.2,8.7,15,300,contact,Ellie,Clear call
2,Spectrogram 1,1,12.1,15.3,20,250,greeting,Dumbo,Group reunion
3,Spectrogram 1,1,22.4,25.1,30,400,alarm,Unknown,Predator nearby?
```

---

## 📋 Method 2: Using Your Existing CSV

You **already have** timestamps in your CSV! Just add label columns:

### Original CSV
```csv
Selection,Sound_file,Start_time,End_time
1,airplane_noise.wav,5.2,8.7
2,vehicle_noise.wav,12.1,15.3
```

### Add Label Columns (in Excel/Google Sheets)

```csv
Selection,Sound_file,Start_time,End_time,CallType,ElephantID,Confidence
1,airplane_noise.wav,5.2,8.7,contact,Ellie,high
2,vehicle_noise.wav,12.1,15.3,greeting,Dumbo,medium
3,generator_noise.wav,22.4,25.1,contact,Ellie,high
```

**Then convert to our format:**

```python
import pandas as pd
import json

# Load your annotated CSV
df = pd.read_csv('annotated_calls.csv')

# Map text labels to numbers
call_type_map = {
    'contact': 0,
    'greeting': 1,
    'alarm': 2,
    'mating': 3
}

elephant_map = {
    'Ellie': 0,
    'Dumbo': 1,
    'Jumbo': 2,
    'Unknown': -1
}

# Create labels dict for cleaned files
labels = {}

for _, row in df.iterrows():
    # Match to your cleaned filename
    selection_id = row['Selection']
    sound_file = row['Sound_file'].replace('.wav', '')
    cleaned_name = f"selection_{selection_id:03d}_{sound_file}_cleaned.wav"
    
    # Add numeric label
    labels[cleaned_name] = call_type_map[row['CallType']]

# Save as JSON
with open('data/labels.json', 'w') as f:
    json.dump(labels, f, indent=2)

print(f"✅ Created labels for {len(labels)} files")
```

**Output: `data/labels.json`**
```json
{
  "selection_001_airplane_noise_cleaned.wav": 0,
  "selection_002_vehicle_noise_cleaned.wav": 1,
  "selection_003_generator_noise_cleaned.wav": 0
}
```

---

## 📋 Method 3: Semi-Automated Labeling Tool

**Best for**: Medium datasets (100-1000 calls), faster annotation

### Using Label Studio (Free, Open-Source)

#### Install Label Studio
```bash
pip install label-studio
label-studio
# Opens browser at http://localhost:8080
```

#### Create Project
1. **Create Project** → "Audio Classification"
2. **Import Data** → Upload your cleaned WAV files
3. **Configure Labeling Interface**:

```xml
<View>
  <Audio name="audio" value="$audio"/>
  <Header value="Listen and classify the elephant call"/>
  <Choices name="call_type" toName="audio" choice="single">
    <Choice value="contact"/>
    <Choice value="greeting"/>
    <Choice value="alarm"/>
    <Choice value="mating"/>
    <Choice value="unknown"/>
  </Choices>
  <Choices name="elephant" toName="audio" choice="single">
    <Choice value="Ellie"/>
    <Choice value="Dumbo"/>
    <Choice value="Jumbo"/>
    <Choice value="Unknown"/>
  </Choices>
  <Number name="confidence" toName="audio" min="1" max="5" step="1"/>
</View>
```

4. **Label**: Play audio → Select labels → Submit
5. **Export**: JSON or CSV format

#### Convert Label Studio Export
```python
import json

# Load Label Studio export
with open('label_studio_export.json') as f:
    annotations = json.load(f)

# Convert to our format
labels = {}
for item in annotations:
    filename = item['file_upload'].split('/')[-1]
    call_type = item['annotations'][0]['result'][0]['value']['choices'][0]
    
    type_map = {'contact': 0, 'greeting': 1, 'alarm': 2, 'mating': 3}
    labels[filename] = type_map[call_type]

# Save
with open('data/labels.json', 'w') as f:
    json.dump(labels, f, indent=2)
```

---

## 📋 Method 4: Active Learning (Smart Semi-Automation)

**Best for**: Large datasets (1000+ calls), limited time

### Workflow

1. **Manually label** ~50-100 calls (10% of data)
2. **Train initial model**
3. **Model predicts** remaining calls
4. **You review** only the **uncertain predictions**
5. **Retrain** with corrected labels
6. **Repeat** until satisfied

### Implementation

```python
from src.ai_training import RumbleTrainer
import numpy as np

# Step 1: Manually label small subset
manually_labeled = {
    "selection_001_cleaned.wav": 0,
    "selection_002_cleaned.wav": 1,
    # ... 50 more
}

# Step 2: Train initial model
trainer = RumbleTrainer('outputs/audio', manually_labeled)
results = trainer.train_random_forest()

# Step 3: Predict on unlabeled data
all_files = list(Path('outputs/audio').glob('*.wav'))
unlabeled_files = [f for f in all_files if f.name not in manually_labeled]

predictions = []
uncertainties = []

for audio_file in unlabeled_files:
    signal, sr = librosa.load(audio_file)
    features = extract_rumble_features(signal, sr)
    vector = features_to_vector(features)
    
    # Get prediction probabilities
    probs = results['model'].predict_proba([vector])[0]
    pred = np.argmax(probs)
    uncertainty = 1 - probs.max()  # Lower confidence = higher uncertainty
    
    predictions.append((audio_file.name, pred, uncertainty))
    uncertainties.append(uncertainty)

# Step 4: Sort by uncertainty (review most uncertain first)
sorted_preds = sorted(predictions, key=lambda x: x[2], reverse=True)

# Step 5: Review top 50 most uncertain
print("Review these predictions (most uncertain first):")
for filename, pred, unc in sorted_preds[:50]:
    print(f"{filename}: Predicted={pred}, Uncertainty={unc:.3f}")
    # Listen to file, correct if wrong, add to labels
```

---

## 📋 Method 5: Weak Supervision (Programmatic Labeling)

**Best for**: Very large datasets, when you have domain knowledge rules

### Using Acoustic Rules

```python
from src.ai_features import extract_rumble_features
import librosa

def programmatic_label(audio_file):
    """
    Label based on acoustic features (weak/noisy labels).
    """
    signal, sr = librosa.load(audio_file)
    features = extract_rumble_features(signal, sr)
    
    # Rule-based labeling
    f0 = features.fundamental_freq
    duration = features.duration
    harmonic_ratio = features.harmonic_energy_ratio
    
    # Contact calls: Low F0, long duration, high harmonicity
    if f0 < 25 and duration > 2.5 and harmonic_ratio > 0.7:
        return 0  # contact
    
    # Greeting: Medium F0, shorter, moderate harmonicity
    elif 25 <= f0 < 40 and 1.0 < duration < 2.5 and harmonic_ratio > 0.6:
        return 1  # greeting
    
    # Alarm: Higher F0, variable duration
    elif f0 >= 40 and harmonic_ratio > 0.5:
        return 2  # alarm
    
    else:
        return -1  # unknown (filter out)

# Apply to all files
weak_labels = {}
for audio_file in Path('outputs/audio').glob('*.wav'):
    label = programmatic_label(audio_file)
    if label != -1:  # Keep only confident labels
        weak_labels[audio_file.name] = label

print(f"Auto-labeled {len(weak_labels)} files")
# Use for pre-training, then fine-tune with manual labels
```

---

## 📋 Method 6: Crowdsourcing

**Best for**: Very large datasets, citizen science projects

### Using Amazon Mechanical Turk or Zooniverse

**Zooniverse Workflow:**
1. Create project at zooniverse.org
2. Upload audio files
3. Create simple interface: "Listen and classify"
4. Multiple volunteers label each call
5. Aggregate labels (majority vote)

**Example task:**
```
Listen to this elephant call.
What type of call is this?
○ Contact call (social bonding)
○ Greeting call (reunion)
○ Alarm call (danger)
○ Mating call
○ Unknown/Unclear

How confident are you?
○ Very confident
○ Somewhat confident
○ Not confident
```

---

## 🎯 Label Quality Best Practices

### 1. **Multiple Annotators**

Have 2-3 people label the same calls:

```python
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Annotator 1 labels
ann1 = [0, 1, 0, 2, 1, 0]

# Annotator 2 labels  
ann2 = [0, 1, 0, 2, 2, 0]

# Measure agreement
kappa = cohen_kappa_score(ann1, ann2)
print(f"Inter-annotator agreement: {kappa:.3f}")
# kappa > 0.8 = good agreement
# kappa < 0.6 = need clearer guidelines

# Use majority vote for final labels
final_labels = []
for i in range(len(ann1)):
    if ann1[i] == ann2[i]:
        final_labels.append(ann1[i])
    else:
        # Disagreement - need 3rd annotator or expert review
        final_labels.append(None)
```

### 2. **Label Guidelines Document**

Create clear definitions:

```markdown
# Elephant Call Labeling Guidelines

## Contact Call (Label: 0)
- **Frequency**: 15-35 Hz (fundamental)
- **Duration**: > 2 seconds
- **Context**: Social bonding, maintaining group cohesion
- **Spectrogram**: Horizontal bands, steady frequency
- **Audio**: Deep rumble, "rrrrrrr"

## Greeting Call (Label: 1)
- **Frequency**: 25-50 Hz
- **Duration**: 1-3 seconds
- **Context**: Reunion after separation
- **Spectrogram**: Rising frequency contour
- **Audio**: Higher pitch rumble, "rrrOOOoo"

## Alarm Call (Label: 2)
- **Frequency**: 40-100 Hz
- **Duration**: 0.5-2 seconds
- **Context**: Predator, danger
- **Spectrogram**: Choppy, multiple bursts
- **Audio**: Sharp, rapid rumbles
```

### 3. **Confidence Scores**

Include uncertainty in labels:

```json
{
  "selection_001_cleaned.wav": {
    "label": 0,
    "confidence": "high",
    "annotator": "researcher_1",
    "notes": "Clear contact call, no ambiguity"
  },
  "selection_002_cleaned.wav": {
    "label": 1,
    "confidence": "low",
    "notes": "Could be greeting or contact, overlapping calls"
  }
}
```

Later, train only on high-confidence labels:

```python
# Load labels with confidence
import json
with open('data/labels_with_confidence.json') as f:
    data = json.load(f)

# Filter to high confidence only
high_conf_labels = {
    k: v['label'] 
    for k, v in data.items() 
    if v.get('confidence') == 'high'
}

# Train on clean data
trainer = RumbleTrainer('outputs/audio', high_conf_labels)
```

---

## 🛠️ Helper Script: Create Labels from Your CSV

Save this as `scripts/create_labels.py`:

```python
#!/usr/bin/env python3
"""
Convert annotated CSV to labels.json format.

Usage:
    python scripts/create_labels.py \
        --csv data/annotated_calls.csv \
        --output data/labels.json \
        --label-column CallType
"""

import pandas as pd
import json
import argparse
from pathlib import Path


def create_labels(csv_path, output_path, label_column, mapping=None):
    """
    Convert CSV annotations to labels.json.
    
    Args:
        csv_path: Path to annotated CSV
        output_path: Where to save labels.json
        label_column: Name of column with labels
        mapping: Dict to map text labels to numbers
    """
    df = pd.read_csv(csv_path)
    
    # Default mapping if not provided
    if mapping is None:
        unique_labels = df[label_column].unique()
        mapping = {label: i for i, label in enumerate(unique_labels)}
        print(f"Auto-created mapping: {mapping}")
    
    labels = {}
    
    for _, row in df.iterrows():
        # Create cleaned filename
        selection_id = row['Selection']
        sound_file = Path(row['Sound_file']).stem
        cleaned_name = f"selection_{selection_id:03d}_{sound_file}_cleaned.wav"
        
        # Map label
        text_label = row[label_column]
        if text_label in mapping:
            labels[cleaned_name] = mapping[text_label]
        else:
            print(f"⚠️  Unknown label '{text_label}' in row {row['Selection']}")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"✅ Created {len(labels)} labels")
    print(f"   Saved to: {output_path}")
    print(f"\nLabel distribution:")
    label_counts = pd.Series(labels.values()).value_counts()
    for label_num, count in label_counts.items():
        label_text = [k for k, v in mapping.items() if v == label_num][0]
        print(f"   {label_text} ({label_num}): {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Annotated CSV file')
    parser.add_argument('--output', default='data/labels.json', help='Output JSON')
    parser.add_argument('--label-column', default='CallType', help='Label column name')
    
    args = parser.parse_args()
    
    # Define your label mapping
    mapping = {
        'contact': 0,
        'greeting': 1,
        'alarm': 2,
        'mating': 3,
        'unknown': -1
    }
    
    create_labels(args.csv, args.output, args.label_column, mapping)
```

**Usage:**
```bash
python scripts/create_labels.py \
    --csv data/my_annotations.csv \
    --output data/labels.json \
    --label-column CallType
```

---

## 📊 Summary: Which Method to Use?

| Dataset Size | Time Available | Method | Accuracy |
|--------------|----------------|--------|----------|
| < 100 calls | Hours | Manual (Raven) | ⭐⭐⭐⭐⭐ Best |
| 100-500 | Days | Label Studio | ⭐⭐⭐⭐ Good |
| 500-1000 | Week | Active Learning | ⭐⭐⭐⭐ Good |
| 1000+ | Limited | Weak Supervision | ⭐⭐⭐ OK (noisy) |
| 1000+ | Unlimited | Crowdsourcing | ⭐⭐⭐⭐ Good |

**Recommendation**: Start with **50-100 manual labels** → Train model → Use **active learning** for the rest.

---

## 🎓 Next Steps

1. Choose your labeling method
2. Create initial labels (50-100 calls)
3. Run the helper script to generate `labels.json`
4. Train your first model!

```python
from src.ai_training import RumbleTrainer

trainer = RumbleTrainer('outputs/audio', 'data/labels.json')
results = trainer.train_random_forest()
```

---

📧 Questions about labeling? Open an issue on GitHub!
