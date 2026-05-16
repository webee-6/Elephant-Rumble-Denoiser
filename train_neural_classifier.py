#!/usr/bin/env python3
"""
Train neural network classifier with windowing and segmentation.

Usage:
    python train_neural_classifier.py --audio outputs/audio --labels data/labels.json
    python train_neural_classifier.py --audio outputs/audio --labels data/labels.json --window 1.0 --overlap 0.5
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.neural_classifier import RumbleClassificationTrainer
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, save_path='models/training_history.png'):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Training history saved to: {save_path}")
    

def main():
    parser = argparse.ArgumentParser(
        description='Train neural network for rumble classification with windowing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python train_neural_classifier.py \\
        --audio outputs/audio \\
        --labels data/labels.json

    # Custom windowing
    python train_neural_classifier.py \\
        --audio outputs/audio \\
        --labels data/labels.json \\
        --window 1.5 \\
        --overlap 0.75 \\
        --epochs 150

    # Fast training (fewer epochs)
    python train_neural_classifier.py \\
        --audio outputs/audio \\
        --labels data/labels.json \\
        --epochs 50 \\
        --batch-size 16
        """
    )
    
    parser.add_argument(
        '--audio',
        required=True,
        help='Directory with cleaned audio files'
    )
    
    parser.add_argument(
        '--labels',
        required=True,
        help='JSON file with labels'
    )
    
    parser.add_argument(
        '--window',
        type=float,
        default=1.0,
        help='Window length in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.5,
        help='Overlap length in seconds (default: 0.5, i.e., 50%% overlap)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size (number of files per batch) (default: 8)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    
    parser.add_argument(
        '--output',
        default='models',
        help='Output directory for models (default: models)'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip plotting training history'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.audio).exists():
        print(f"❌ Audio directory not found: {args.audio}")
        sys.exit(1)
    
    if not Path(args.labels).exists():
        print(f"❌ Labels file not found: {args.labels}")
        sys.exit(1)
    
    # Print configuration
    print("="*70)
    print("🧠 NEURAL NETWORK RUMBLE CLASSIFIER")
    print("="*70)
    print(f"Audio directory:  {args.audio}")
    print(f"Labels file:      {args.labels}")
    print(f"Window size:      {args.window}s")
    print(f"Hop size:         {args.overlap}s (overlap: {(args.window - args.overlap)/args.window*100:.0f}%)")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Learning rate:    {args.learning_rate}")
    print(f"Output directory: {args.output}")
    print("="*70)
    
    # Create trainer
    trainer = RumbleClassificationTrainer(
        audio_dir=args.audio,
        labels_file=args.labels,
        window_length_sec=args.window,
        hop_length_sec=args.overlap,
        output_dir=args.output
    )
    
    # Train
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    model = trainer.train(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        early_stopping_patience=15
    )
    
    # Plot results
    if not args.no_plot:
        plot_training_history(trainer.history, save_path=f"{args.output}/training_history.png")
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"✅ Model saved to: {args.output}/best_model.pth")
    print(f"✅ Final validation accuracy: {max(trainer.history['val_acc']):.4f}")
    
    # Test prediction on first file
    print("\n" + "="*70)
    print("TESTING PREDICTION")
    print("="*70)
    
    test_file = list(Path(args.audio).glob('*.wav'))[0]
    result = trainer.predict(str(test_file))
    
    print(f"Test file: {test_file.name}")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Probabilities: {[f'{p:.3f}' for p in result['probabilities']]}")
    print(f"Number of windows: {result['num_windows']}")
    
    if result['attention_weights']:
        print(f"\nAttention weights (focus on windows):")
        attn = np.array(result['attention_weights'])
        top_5 = np.argsort(attn)[-5:][::-1]
        for i, idx in enumerate(top_5):
            print(f"  {i+1}. Window {idx}: {attn[idx]:.4f}")
    
    print("\n🎉 All done! Use the trained model with:")
    print(f"   from src.neural_classifier import RumbleClassificationTrainer")
    print(f"   trainer = RumbleClassificationTrainer('{args.audio}', '{args.labels}')")
    print(f"   result = trainer.predict('path/to/new_rumble.wav')")


if __name__ == "__main__":
    main()
