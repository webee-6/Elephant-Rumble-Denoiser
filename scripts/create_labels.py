#!/usr/bin/env python3
"""
Convert annotated CSV to labels.json format for machine learning.

Usage:
    python scripts/create_labels.py --csv data/annotated_calls.csv --output data/labels.json
    python scripts/create_labels.py --csv data/annotated_calls.csv --label-column CallType --mapping contact:0,greeting:1,alarm:2
"""

import pandas as pd
import json
import argparse
from pathlib import Path
import sys


def create_labels(csv_path: str,
                 output_path: str,
                 label_column: str = 'CallType',
                 mapping: dict = None,
                 selection_column: str = 'Selection',
                 filename_column: str = 'Sound_file'):
    """
    Convert CSV annotations to labels.json.
    
    Args:
        csv_path: Path to annotated CSV
        output_path: Where to save labels.json
        label_column: Name of column with labels
        mapping: Dict to map text labels to numbers
        selection_column: Column with selection ID
        filename_column: Column with original filename
    """
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded CSV: {len(df)} rows")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        sys.exit(1)
    
    # Check required columns
    required_cols = [selection_column, filename_column]
    if label_column:
        required_cols.append(label_column)
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        print(f"   Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Auto-create mapping if not provided
    if mapping is None and label_column:
        unique_labels = sorted(df[label_column].dropna().unique())
        mapping = {label: i for i, label in enumerate(unique_labels)}
        print(f"\n📊 Auto-created label mapping:")
        for text, num in mapping.items():
            print(f"   {text} → {num}")
    
    # Create labels dict
    labels = {}
    skipped = []
    
    for idx, row in df.iterrows():
        try:
            # Get selection ID
            selection_id = int(row[selection_column])
            
            # Get original filename and clean it
            sound_file = str(row[filename_column])
            sound_file_base = Path(sound_file).stem
            
            # Sanitize filename (same as pipeline)
            import re
            sound_file_base = sound_file_base.replace(' ', '_')
            sound_file_base = re.sub(r'[^\w\-.]', '_', sound_file_base)
            sound_file_base = re.sub(r'_+', '_', sound_file_base).strip('_')
            
            # Create cleaned filename
            cleaned_name = f"selection_{selection_id:03d}_{sound_file_base}_cleaned.wav"
            
            # Get label
            if label_column and label_column in row:
                text_label = row[label_column]
                
                # Handle NaN/empty
                if pd.isna(text_label) or text_label == '':
                    skipped.append((selection_id, "Empty label"))
                    continue
                
                # Map to number
                if mapping and text_label in mapping:
                    labels[cleaned_name] = mapping[text_label]
                elif mapping:
                    skipped.append((selection_id, f"Unknown label: {text_label}"))
                else:
                    # No mapping, use raw value
                    labels[cleaned_name] = text_label
            else:
                # No label column, just create mapping
                labels[cleaned_name] = 0  # Default
        
        except Exception as e:
            skipped.append((idx, str(e)))
    
    # Report skipped
    if skipped:
        print(f"\n⚠️  Skipped {len(skipped)} rows:")
        for sel_id, reason in skipped[:10]:  # Show first 10
            print(f"   Row {sel_id}: {reason}")
        if len(skipped) > 10:
            print(f"   ... and {len(skipped)-10} more")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2, sort_keys=True)
    
    print(f"\n✅ Created labels for {len(labels)} files")
    print(f"   Saved to: {output_path}")
    
    # Show distribution
    if mapping and labels:
        print(f"\n📊 Label distribution:")
        from collections import Counter
        counts = Counter(labels.values())
        
        for label_num in sorted(counts.keys()):
            # Find text label
            label_text = [k for k, v in mapping.items() if v == label_num]
            text = label_text[0] if label_text else f"Label_{label_num}"
            count = counts[label_num]
            percentage = count / len(labels) * 100
            print(f"   {text:15s} ({label_num}): {count:4d} ({percentage:5.1f}%)")
    
    return labels


def main():
    parser = argparse.ArgumentParser(
        description='Convert annotated CSV to labels.json for ML training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-detect labels
    python scripts/create_labels.py --csv data/annotations.csv

    # Custom mapping
    python scripts/create_labels.py \\
        --csv data/annotations.csv \\
        --mapping "contact:0,greeting:1,alarm:2,mating:3"
    
    # Different column names
    python scripts/create_labels.py \\
        --csv data/annotations.csv \\
        --label-column Type \\
        --selection-column ID \\
        --filename-column File
        """
    )
    
    parser.add_argument(
        '--csv',
        required=True,
        help='Path to annotated CSV file'
    )
    
    parser.add_argument(
        '--output',
        default='data/labels.json',
        help='Output JSON file path (default: data/labels.json)'
    )
    
    parser.add_argument(
        '--label-column',
        default='CallType',
        help='Name of column with labels (default: CallType)'
    )
    
    parser.add_argument(
        '--selection-column',
        default='Selection',
        help='Name of column with selection ID (default: Selection)'
    )
    
    parser.add_argument(
        '--filename-column',
        default='Sound_file',
        help='Name of column with filename (default: Sound_file)'
    )
    
    parser.add_argument(
        '--mapping',
        help='Label mapping as "text1:num1,text2:num2" (e.g., "contact:0,greeting:1")'
    )
    
    args = parser.parse_args()
    
    # Parse mapping if provided
    mapping = None
    if args.mapping:
        try:
            pairs = args.mapping.split(',')
            mapping = {}
            for pair in pairs:
                text, num = pair.split(':')
                mapping[text.strip()] = int(num.strip())
        except Exception as e:
            print(f"❌ Error parsing mapping: {e}")
            print("   Format: 'label1:0,label2:1,label3:2'")
            sys.exit(1)
    
    # Create labels
    create_labels(
        csv_path=args.csv,
        output_path=args.output,
        label_column=args.label_column,
        mapping=mapping,
        selection_column=args.selection_column,
        filename_column=args.filename_column
    )


if __name__ == "__main__":
    main()
