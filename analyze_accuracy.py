#!/usr/bin/env python
"""Analyze model accuracy and anomaly detection performance."""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report

# Read scores
scores_df = pd.read_csv('scores.csv')

print('='*80)
print('MODEL ACCURACY & ANOMALY DETECTION ANALYSIS')
print('='*80)

# Check for labeled data
background_events = scores_df[scores_df['label'] == 0]
signal_events = scores_df[scores_df['label'] == 1]

print(f'\nData Distribution:')
print(f'  Background Events (label=0): {len(background_events):,} ({len(background_events)/len(scores_df)*100:.1f}%)')
print(f'  Signal Events (label=1): {len(signal_events):,} ({len(signal_events)/len(scores_df)*100:.1f}%)')

# Reconstruction error statistics by class
print(f'\n--- Reconstruction Error by Event Type ---')
print(f'Background Events (Normal):')
print(f'  Mean Score: {background_events["anomaly_score"].mean():.6f}')
print(f'  Median Score: {background_events["anomaly_score"].median():.6f}')
print(f'  Std Dev: {background_events["anomaly_score"].std():.6f}')
print(f'  Min: {background_events["anomaly_score"].min():.6f}')
print(f'  Max: {background_events["anomaly_score"].max():.6f}')

if len(signal_events) > 0:
    print(f'\nSignal Events (Anomalous):')
    print(f'  Mean Score: {signal_events["anomaly_score"].mean():.6f}')
    print(f'  Median Score: {signal_events["anomaly_score"].median():.6f}')
    print(f'  Std Dev: {signal_events["anomaly_score"].std():.6f}')
    print(f'  Min: {signal_events["anomaly_score"].min():.6f}')
    print(f'  Max: {signal_events["anomaly_score"].max():.6f}')
    
    # Calculate AUC-ROC
    y_true = (scores_df['label'] == 1).astype(int)
    y_scores = scores_df['anomaly_score'].values
    auc = roc_auc_score(y_true, y_scores)
    
    print(f'\n--- Performance Metrics ---')
    print(f'AUC-ROC Score: {auc:.4f}')
    
    # Find optimal threshold (Youden's J statistic)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    print(f'Optimal Threshold (Youden): {best_threshold:.6f}')
    print(f'  True Positive Rate: {tpr[best_idx]:.4f}')
    print(f'  False Positive Rate: {fpr[best_idx]:.4f}')
    
    # Confusion matrix at optimal threshold
    y_pred = (y_scores > best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    print(f'\n--- Confusion Matrix (at threshold={best_threshold:.6f}) ---')
    print(f'True Positives (Detected Anomalies): {tp:,}')
    print(f'True Negatives (Correctly Identified Normal): {tn:,}')
    print(f'False Positives (False Alarms): {fp:,}')
    print(f'False Negatives (Missed Anomalies): {fn:,}')
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    print(f'\n--- Classification Metrics ---')
    print(f'Sensitivity (Recall): {sensitivity:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'F1-Score: {f1:.4f}')
else:
    print(f'\nNote: No labeled signal events found in dataset.')
    print(f'Model was trained unsupervised on background events.')
    print(f'Anomaly scores represent reconstruction error (MSE).')
    print(f'Higher scores indicate more anomalous events.')

# Score distribution analysis
print(f'\n--- Score Distribution Percentiles ---')
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    val = np.percentile(scores_df['anomaly_score'], p)
    print(f'  {p}th percentile: {val:.6f}')

print('\n' + '='*80)
