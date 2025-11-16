#!/usr/bin/env python3
"""
Heat Stress Gene Prediction Script

This script loads pre-trained machine learning models to predict whether a gene
is heat-stress-responsive based on its biological features.

Usage:
    # Single gene prediction
    python predict_gene.py --log2FC 2.3 --neg_log10_pvalue 5.1 --baseMean 600 --gene_length 1500 --GC_content 0.42

    # Batch prediction from CSV
    python predict_gene.py --input genes.csv --output predictions.csv
"""

import argparse
import json
import sys
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np


class GenePredictor:
    """Load models and make predictions on gene data."""

    def __init__(self, models_dir='models'):
        """
        Initialize predictor by loading models, scalers, and feature info.

        Args:
            models_dir: Directory containing model artifacts
        """
        self.models_dir = Path(models_dir)

        # Check if models directory exists
        if not self.models_dir.exists():
            raise FileNotFoundError(
                f"Models directory not found: {self.models_dir}\n"
                f"Please ensure model artifacts are in the '{models_dir}' folder."
            )

        print("Loading models...")

        # Load models
        self.rf_model = self._load_artifact('rf_model.pkl')
        self.lr_model = self._load_artifact('lr_model.pkl')
        self.svm_model = self._load_artifact('svm_model.pkl')

        # Load scalers
        self.scaler_lr = self._load_artifact('scaler_lr.pkl')
        self.scaler_svm = self._load_artifact('scaler_svm.pkl')

        # Load feature columns
        feature_file = self.models_dir / 'feature_columns.json'
        if not feature_file.exists():
            raise FileNotFoundError(f"Feature columns file not found: {feature_file}")

        with open(feature_file, 'r') as f:
            self.feature_columns = json.load(f)

        print(f"[OK] Models loaded successfully")
        print(f"[OK] Expected features: {self.feature_columns}\n")

    def _load_artifact(self, filename):
        """Load a single model artifact with error handling."""
        filepath = self.models_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)

    def validate_features(self, features_dict):
        """
        Validate that input features are correct and within reasonable ranges.

        Args:
            features_dict: Dictionary of feature values

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check all required features are present
        missing = set(self.feature_columns) - set(features_dict.keys())
        if missing:
            return False, f"Missing features: {missing}"

        # Check for extra features
        extra = set(features_dict.keys()) - set(self.feature_columns)
        if extra:
            return False, f"Unknown features: {extra}"

        # Validate ranges
        validations = {
            'log2FC': (-10, 10, "log2 fold change should be between -10 and 10"),
            'neg_log10_pvalue': (0, 50, "negative log10 p-value should be between 0 and 50"),
            'baseMean': (0, 100000, "baseMean should be between 0 and 100,000"),
            'gene_length': (100, 20000, "gene_length should be between 100 and 20,000 bp"),
            'GC_content': (0, 1, "GC_content should be between 0 and 1")
        }

        for feature, (min_val, max_val, msg) in validations.items():
            value = features_dict[feature]
            if not isinstance(value, (int, float)):
                return False, f"{feature} must be numeric, got {type(value).__name__}"
            if not (min_val <= value <= max_val):
                return False, f"Invalid {feature}={value}: {msg}"

        return True, None

    def predict_single(self, features_dict, verbose=True):
        """
        Predict for a single gene.

        Args:
            features_dict: Dictionary with feature values
            verbose: Whether to print detailed output

        Returns:
            Dictionary with predictions from all models
        """
        # Validate features
        is_valid, error = self.validate_features(features_dict)
        if not is_valid:
            raise ValueError(error)

        # Create DataFrame with correct column order
        features_df = pd.DataFrame([features_dict])[self.feature_columns]

        if verbose:
            print("Input Features:")
            for col in self.feature_columns:
                print(f"  {col}: {features_dict[col]}")
            print()

        # RandomForest prediction (no scaling)
        rf_pred = self.rf_model.predict(features_df)[0]
        rf_prob = self.rf_model.predict_proba(features_df)[0, 1]

        # LogisticRegression prediction (with scaling)
        features_scaled_lr = self.scaler_lr.transform(features_df)
        lr_pred = self.lr_model.predict(features_scaled_lr)[0]
        lr_prob = self.lr_model.predict_proba(features_scaled_lr)[0, 1]

        # SVM prediction (with scaling)
        features_scaled_svm = self.scaler_svm.transform(features_df)
        svm_pred = self.svm_model.predict(features_scaled_svm)[0]
        svm_prob = self.svm_model.predict_proba(features_scaled_svm)[0, 1]

        # Ensemble prediction (average probabilities)
        ensemble_prob = (rf_prob + lr_prob + svm_prob) / 3
        ensemble_pred = 1 if ensemble_prob > 0.5 else 0

        results = {
            'RandomForest': {'prediction': rf_pred, 'probability': rf_prob},
            'LogisticRegression': {'prediction': lr_pred, 'probability': lr_prob},
            'SVM': {'prediction': svm_pred, 'probability': svm_prob},
            'Ensemble': {'prediction': ensemble_pred, 'probability': ensemble_prob}
        }

        if verbose:
            self._print_predictions(results)

        return results

    def _print_predictions(self, results):
        """Print formatted prediction results."""
        print("=" * 70)
        print("PREDICTION RESULTS")
        print("=" * 70)

        for model_name, result in results.items():
            pred = result['prediction']
            prob = result['probability']
            label = "RESPONSIVE" if pred == 1 else "Non-responsive"
            confidence = prob if pred == 1 else (1 - prob)

            if model_name == 'Ensemble':
                print("\n" + "-" * 70)
                print(f">>> {model_name.upper()} PREDICTION <<<")
            else:
                print(f"\n{model_name}:")

            print(f"  Prediction: {label}")
            print(f"  Probability (responsive): {prob:.4f}")
            print(f"  Confidence: {confidence:.2%}")

        print("\n" + "=" * 70)

    def predict_batch(self, input_file, output_file=None, verbose=True):
        """
        Predict for multiple genes from a CSV file.

        Args:
            input_file: Path to input CSV with gene features
            output_file: Path to save predictions (optional)
            verbose: Whether to print progress

        Returns:
            DataFrame with predictions
        """
        if verbose:
            print(f"Loading genes from: {input_file}")

        # Load input data
        try:
            genes_df = pd.read_csv(input_file)
        except Exception as e:
            raise ValueError(f"Error reading input file: {e}")

        if verbose:
            print(f"Loaded {len(genes_df)} genes\n")

        # Validate columns
        missing = set(self.feature_columns) - set(genes_df.columns)
        if missing:
            raise ValueError(f"Input CSV missing required columns: {missing}")

        # Extract features in correct order
        features_df = genes_df[self.feature_columns]

        # Validate all rows
        for idx, row in features_df.iterrows():
            is_valid, error = self.validate_features(row.to_dict())
            if not is_valid:
                raise ValueError(f"Row {idx}: {error}")

        if verbose:
            print("Making predictions...")

        # RandomForest predictions
        rf_pred = self.rf_model.predict(features_df)
        rf_prob = self.rf_model.predict_proba(features_df)[:, 1]

        # LogisticRegression predictions
        features_scaled_lr = self.scaler_lr.transform(features_df)
        lr_pred = self.lr_model.predict(features_scaled_lr)
        lr_prob = self.lr_model.predict_proba(features_scaled_lr)[:, 1]

        # SVM predictions
        features_scaled_svm = self.scaler_svm.transform(features_df)
        svm_pred = self.svm_model.predict(features_scaled_svm)
        svm_prob = self.svm_model.predict_proba(features_scaled_svm)[:, 1]

        # Ensemble predictions
        ensemble_prob = (rf_prob + lr_prob + svm_prob) / 3
        ensemble_pred = (ensemble_prob > 0.5).astype(int)

        # Create results DataFrame
        results_df = genes_df.copy()

        # Add predictions
        results_df['RF_prediction'] = rf_pred
        results_df['RF_probability'] = rf_prob
        results_df['LR_prediction'] = lr_pred
        results_df['LR_probability'] = lr_prob
        results_df['SVM_prediction'] = svm_pred
        results_df['SVM_probability'] = svm_prob
        results_df['Ensemble_prediction'] = ensemble_pred
        results_df['Ensemble_probability'] = ensemble_prob

        # Add human-readable labels
        results_df['RF_label'] = results_df['RF_prediction'].map({0: 'Non-responsive', 1: 'Responsive'})
        results_df['LR_label'] = results_df['LR_prediction'].map({0: 'Non-responsive', 1: 'Responsive'})
        results_df['SVM_label'] = results_df['SVM_prediction'].map({0: 'Non-responsive', 1: 'Responsive'})
        results_df['Ensemble_label'] = results_df['Ensemble_prediction'].map({0: 'Non-responsive', 1: 'Responsive'})

        # Sort by ensemble probability (descending)
        results_df = results_df.sort_values('Ensemble_probability', ascending=False).reset_index(drop=True)

        if verbose:
            print(f"[OK] Predictions complete\n")
            print("Summary:")
            print(f"  Responsive genes (Ensemble): {ensemble_pred.sum()} ({ensemble_pred.sum()/len(ensemble_pred)*100:.1f}%)")
            print(f"  Non-responsive genes: {(1-ensemble_pred).sum()} ({(1-ensemble_pred).sum()/len(ensemble_pred)*100:.1f}%)")
            print()

        # Save to file if specified
        if output_file:
            results_df.to_csv(output_file, index=False)
            if verbose:
                print(f"[OK] Results saved to: {output_file}\n")

        return results_df


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Predict heat-stress-responsive genes using trained ML models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single gene prediction
  python predict_gene.py --log2FC 2.3 --neg_log10_pvalue 5.1 --baseMean 600 --gene_length 1500 --GC_content 0.42

  # Batch prediction
  python predict_gene.py --input genes.csv --output predictions.csv

  # Use models from different directory
  python predict_gene.py --models_dir ../models --log2FC 2.0 --neg_log10_pvalue 4.0 --baseMean 500 --gene_length 2000 --GC_content 0.45
        """
    )

    # Model directory
    parser.add_argument('--models_dir', type=str, default='models',
                        help='Directory containing model artifacts (default: models)')

    # Single gene prediction arguments
    parser.add_argument('--log2FC', type=float,
                        help='Log2 fold change (e.g., 2.3)')
    parser.add_argument('--neg_log10_pvalue', type=float,
                        help='Negative log10 of p-value (e.g., 5.1)')
    parser.add_argument('--baseMean', type=float,
                        help='Average expression level (e.g., 600)')
    parser.add_argument('--gene_length', type=int,
                        help='Gene length in base pairs (e.g., 1500)')
    parser.add_argument('--GC_content', type=float,
                        help='GC content as decimal (e.g., 0.42)')

    # Batch prediction arguments
    parser.add_argument('--input', type=str,
                        help='Input CSV file with gene features')
    parser.add_argument('--output', type=str,
                        help='Output CSV file for predictions')

    args = parser.parse_args()

    # Determine mode
    single_mode_args = [args.log2FC, args.neg_log10_pvalue, args.baseMean,
                        args.gene_length, args.GC_content]
    batch_mode = args.input is not None
    single_mode = any(arg is not None for arg in single_mode_args)

    if not batch_mode and not single_mode:
        parser.print_help()
        print("\nError: Please provide either single gene features or --input file")
        sys.exit(1)

    if batch_mode and single_mode:
        print("Error: Cannot use both single gene mode and batch mode simultaneously")
        sys.exit(1)

    # Initialize predictor
    try:
        predictor = GenePredictor(models_dir=args.models_dir)
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

    # Single gene prediction
    if single_mode:
        # Check all features are provided
        if not all(arg is not None for arg in single_mode_args):
            print("Error: For single gene prediction, all features must be provided:")
            print("  --log2FC, --neg_log10_pvalue, --baseMean, --gene_length, --GC_content")
            sys.exit(1)

        features = {
            'log2FC': args.log2FC,
            'neg_log10_pvalue': args.neg_log10_pvalue,
            'baseMean': args.baseMean,
            'gene_length': args.gene_length,
            'GC_content': args.GC_content
        }

        try:
            predictor.predict_single(features, verbose=True)
        except Exception as e:
            print(f"Error making prediction: {e}")
            sys.exit(1)

    # Batch prediction
    else:
        try:
            predictor.predict_batch(args.input, args.output, verbose=True)
        except Exception as e:
            print(f"Error in batch prediction: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
