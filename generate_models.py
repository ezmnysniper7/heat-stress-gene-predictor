#!/usr/bin/env python3
"""
Generate ML models for Heat Stress Gene Predictor

This script trains all 3 models and saves them to disk.
Based on ML_HEAT_STRESS_NOTEBOOK.ipynb
"""

import numpy as np
import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

print("="*70)
print("HEAT STRESS GENE PREDICTION - MODEL TRAINING")
print("="*70)
print()

# Set random seed
np.random.seed(42)

# Generate synthetic data
print("[1/6] Generating synthetic gene dataset...")
n_genes = 2000
n_responsive = int(n_genes * 0.27)
n_non_responsive = n_genes - n_responsive

# Generate log2FC
log2FC_responsive = np.random.normal(loc=2.0, scale=0.8, size=n_responsive)
log2FC_non_responsive = np.random.normal(loc=0.0, scale=0.5, size=n_non_responsive)
log2FC = np.concatenate([log2FC_responsive, log2FC_non_responsive])

# Generate p_value
p_value_responsive = np.random.beta(a=0.5, b=5, size=n_responsive) * 0.05
p_value_non_responsive = np.random.uniform(low=0.05, high=1.0, size=n_non_responsive)
p_value = np.concatenate([p_value_responsive, p_value_non_responsive])
p_value = np.clip(p_value, 1e-10, 1.0)
neg_log10_pvalue = -np.log10(p_value)

# Generate baseMean
baseMean = np.random.uniform(low=5, high=3000, size=n_genes)

# Generate gene_length
gene_length = np.random.randint(low=500, high=5001, size=n_genes)

# Generate GC_content
GC_content = np.random.uniform(low=0.30, high=0.65, size=n_genes)

# Shuffle
shuffle_idx = np.random.permutation(n_genes)
log2FC = log2FC[shuffle_idx]
p_value = p_value[shuffle_idx]
neg_log10_pvalue = neg_log10_pvalue[shuffle_idx]
baseMean = baseMean[shuffle_idx]
gene_length = gene_length[shuffle_idx]
GC_content = GC_content[shuffle_idx]

# Create labels
labels = ((log2FC > 1.0) & (p_value < 0.05)).astype(int)

print(f"   Generated {n_genes} genes")
print(f"   Responsive: {labels.sum()} ({labels.sum()/len(labels)*100:.1f}%)")
print(f"   Non-responsive: {(1-labels).sum()} ({(1-labels).sum()/len(labels)*100:.1f}%)")
print()

# Create features DataFrame
features_df = pd.DataFrame({
    'log2FC': log2FC,
    'neg_log10_pvalue': neg_log10_pvalue,
    'baseMean': baseMean,
    'gene_length': gene_length,
    'GC_content': GC_content
})

feature_columns = features_df.columns.tolist()

# Split data
print("[2/6] Splitting data into train/test sets...")
X = features_df
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print(f"   Training set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")
print()

# Prepare scalers
print("[3/6] Preparing feature scalers...")
scaler_lr = StandardScaler()
scaler_svm = StandardScaler()

X_train_scaled_lr = scaler_lr.fit_transform(X_train)
X_test_scaled_lr = scaler_lr.transform(X_test)

X_train_scaled_svm = scaler_svm.fit_transform(X_train)
X_test_scaled_svm = scaler_svm.transform(X_test)

print("   [OK] Scalers fitted")
print()

# Train models
print("[4/6] Training machine learning models...")
print()

# RandomForest
print("   [1/3] Training RandomForest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_score = rf_model.score(X_test, y_test)
print(f"         Accuracy: {rf_score*100:.2f}%")

# LogisticRegression
print("   [2/3] Training LogisticRegression...")
lr_model = LogisticRegression(
    solver='liblinear',
    random_state=42,
    max_iter=1000
)
lr_model.fit(X_train_scaled_lr, y_train)
lr_score = lr_model.score(X_test_scaled_lr, y_test)
print(f"         Accuracy: {lr_score*100:.2f}%")

# SVM
print("   [3/3] Training SVM...")
svm_model = SVC(
    kernel='rbf',
    probability=True,
    random_state=42,
    gamma='scale'
)
svm_model.fit(X_train_scaled_svm, y_train)
svm_score = svm_model.score(X_test_scaled_svm, y_test)
print(f"         Accuracy: {svm_score*100:.2f}%")
print()

# Save models
print("[5/6] Saving model artifacts...")
joblib.dump(rf_model, 'rf_model.pkl')
print("   [OK] Saved: rf_model.pkl")

joblib.dump(lr_model, 'lr_model.pkl')
print("   [OK] Saved: lr_model.pkl")

joblib.dump(svm_model, 'svm_model.pkl')
print("   [OK] Saved: svm_model.pkl")

joblib.dump(scaler_lr, 'scaler_lr.pkl')
print("   [OK] Saved: scaler_lr.pkl")

joblib.dump(scaler_svm, 'scaler_svm.pkl')
print("   [OK] Saved: scaler_svm.pkl")

with open('feature_columns.json', 'w') as f:
    json.dump(feature_columns, f, indent=2)
print("   [OK] Saved: feature_columns.json")
print()

# Verify
print("[6/6] Verifying saved models...")
try:
    rf_loaded = joblib.load('rf_model.pkl')
    lr_loaded = joblib.load('lr_model.pkl')
    svm_loaded = joblib.load('svm_model.pkl')
    scaler_lr_loaded = joblib.load('scaler_lr.pkl')
    scaler_svm_loaded = joblib.load('scaler_svm.pkl')
    with open('feature_columns.json', 'r') as f:
        features_loaded = json.load(f)
    print("   [OK] All models loaded successfully")
except Exception as e:
    print(f"   [ERROR] Error loading models: {e}")
    exit(1)

print()
print("="*70)
print("SUCCESS! All models trained and saved.")
print("="*70)
print()
print("Model Performance Summary:")
print(f"  RandomForest:        {rf_score*100:.2f}%")
print(f"  LogisticRegression:  {lr_score*100:.2f}%")
print(f"  SVM:                 {svm_score*100:.2f}%")
print()
print("Files created:")
print("  [OK] rf_model.pkl")
print("  [OK] lr_model.pkl")
print("  [OK] svm_model.pkl")
print("  [OK] scaler_lr.pkl")
print("  [OK] scaler_svm.pkl")
print("  [OK] feature_columns.json")
print()
print("Next steps:")
print("  1. Run: cd heat_stress_predictor && python setup_models.py")
print("  2. Launch GUI: python gene_predictor_gui.py")
print("  3. Or try CLI: python predict_gene.py --help")
print()
