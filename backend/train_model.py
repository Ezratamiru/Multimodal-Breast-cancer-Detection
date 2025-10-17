#!/usr/bin/env python3
"""
Mammogram Classification Model Training Script
"""

import os
import sys
import argparse
from ml_model import MammogramClassifier

def main():
    parser = argparse.ArgumentParser(description='Train mammogram classification model')
    parser.add_argument('--data-dir', default='Mammogram-20250924T132236Z-1-001/Mammogram', 
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--img-size', type=int, nargs=2, default=[224, 224], 
                       help='Image size (height width)')
    parser.add_argument('--output', default='mammogram_classifier.h5', 
                       help='Output model file')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found!")
        sys.exit(1)
    
    print("=" * 60)
    print("MAMMOGRAM CLASSIFICATION MODEL TRAINING")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print(f"Output model: {args.output}")
    print("=" * 60)
    
    # Initialize classifier
    classifier = MammogramClassifier(
        img_size=tuple(args.img_size), 
        batch_size=args.batch_size
    )
    
    try:
        # Train the model
        print("\n🚀 Starting training...")
        history = classifier.train(args.data_dir, epochs=args.epochs)
        
        # Evaluate the model
        print("\n📊 Evaluating model...")
        results = classifier.evaluate(args.data_dir)
        
        # Save the model
        print(f"\n💾 Saving model to {args.output}...")
        classifier.save_model(args.output)
        
        # Plot training history
        print("\n📈 Generating training plots...")
        classifier.plot_training_history()
        
        # Print final results
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 60)
        print(f"Final Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Final Test Precision: {results['test_precision']:.4f}")
        print(f"Final Test Recall: {results['test_recall']:.4f}")
        print(f"Final F1-Score: {results['f1_score']:.4f}")
        print(f"Model saved to: {args.output}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
