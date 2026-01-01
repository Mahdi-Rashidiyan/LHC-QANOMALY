#!/usr/bin/env python
"""Analyze precision-recall trade-offs and optimize threshold."""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# Read scores
scores_df = pd.read_csv('scores.csv')

y_true = (scores_df['label'] == 1).astype(int)
y_scores = scores_df['anomaly_score'].values

print('='*80)
print('PRECISION-RECALL TRADE-OFF ANALYSIS')
print('='*80)

# Test specific thresholds
test_thresholds = [0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00]

print('\n--- Performance at Different Thresholds ---\n')
print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Flagged':<12}")
print('-'*60)

results = []

for threshold in test_thresholds:
    y_pred = (y_scores > threshold).astype(int)
    num_flagged = y_pred.sum()
    
    if num_flagged == 0:
        precision = 0
        recall = 0
        f1 = 0
        tp = fp = fn = 0
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    results.append({
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'flagged': num_flagged
    })
    
    print(f"{threshold:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {num_flagged:<12,}")

print('\n' + '='*80)
print('KEY FINDINGS')
print('='*80)

# Find best F1
best_f1_idx = np.argmax([r['f1'] for r in results])
best_f1 = results[best_f1_idx]
print(f"\nBest F1-Score: {best_f1['f1']:.4f}")
print(f"  Threshold: {best_f1['threshold']:.4f}")
print(f"  Precision: {best_f1['precision']:.4f}")
print(f"  Recall: {best_f1['recall']:.4f}")
print(f"  Events Flagged: {best_f1['flagged']:,}")

# Find best precision
best_prec_idx = np.argmax([r['precision'] for r in results])
best_prec = results[best_prec_idx]
print(f"\nHighest Precision: {best_prec['precision']:.4f}")
print(f"  Threshold: {best_prec['threshold']:.4f}")
print(f"  Recall: {best_prec['recall']:.4f}")
print(f"  F1-Score: {best_prec['f1']:.4f}")
print(f"  Events Flagged: {best_prec['flagged']:,}")

print('\n' + '='*80)
print('RECOMMENDATIONS TO IMPROVE MODEL')
print('='*80)
print('''
ISSUE: The current model was trained for only 5 epochs on CPU, which limits
       its learning capacity. Precision is low due to reconstruction error overlap.

SOLUTION: Retrain the model with better hyperparameters

  1. INCREASE TRAINING EPOCHS
     - From 5 → 50 epochs (10x more training)
     - Better convergence on training loss
     
  2. IMPROVE ARCHITECTURE
     - Add regularization (dropout)
     - Increase model capacity with more hidden units
     - Use batch normalization
     
  3. HANDLE CLASS IMBALANCE
     - Use weighted loss function (higher weight for signal class)
     - Adjust batch composition during training
     
  4. OPTIMIZE TRAINING
     - Use validation set to monitor overfitting
     - Early stopping if validation loss plateaus
     - Learning rate scheduling

To retrain, run:
  $ cd c:\\Users\\Liver\\OneDrive\\Desktop\\LHC\\lhc-qanomaly
  $ lhc_qanomaly train  # Will use updated config.py
  
Then re-score and re-evaluate.
''')
print('='*80)
