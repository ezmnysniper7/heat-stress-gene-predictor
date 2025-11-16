#!/usr/bin/env python3
"""
Heat Stress Gene Predictor - GUI Application

A graphical user interface for predicting heat-stress-responsive genes
using pre-trained machine learning models.

Usage:
    python gene_predictor_gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
from pathlib import Path
import joblib
import pandas as pd
import sys


class GenePredictorGUI:
    """Tkinter-based GUI for gene prediction."""

    def __init__(self, root, models_dir='models'):
        """
        Initialize the GUI application.

        Args:
            root: Tkinter root window
            models_dir: Directory containing model artifacts
        """
        self.root = root
        self.root.title("Heat Stress Gene Predictor")
        self.root.geometry("700x800")
        self.root.resizable(False, False)

        # Set color scheme
        self.color_responsive = "#ff6b6b"  # Red for responsive
        self.color_non_responsive = "#51cf66"  # Green for non-responsive
        self.color_neutral = "#e9ecef"  # Gray for neutral

        # Load models
        self.models_dir = Path(models_dir)
        self.load_models()

        # Create GUI elements
        self.create_widgets()

    def load_models(self):
        """Load all model artifacts."""
        try:
            # Load models
            self.rf_model = joblib.load(self.models_dir / 'rf_model.pkl')
            self.lr_model = joblib.load(self.models_dir / 'lr_model.pkl')
            self.svm_model = joblib.load(self.models_dir / 'svm_model.pkl')

            # Load scalers
            self.scaler_lr = joblib.load(self.models_dir / 'scaler_lr.pkl')
            self.scaler_svm = joblib.load(self.models_dir / 'scaler_svm.pkl')

            # Load feature columns
            with open(self.models_dir / 'feature_columns.json', 'r') as f:
                self.feature_columns = json.load(f)

            print("[OK] Models loaded successfully")

        except Exception as e:
            messagebox.showerror(
                "Error Loading Models",
                f"Failed to load models from '{self.models_dir}':\n\n{str(e)}\n\n"
                f"Please ensure all model files are present."
            )
            sys.exit(1)

    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Heat Stress Gene Predictor",
            font=("Arial", 18, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 5))

        # Creator credit
        credit_label = ttk.Label(
            main_frame,
            text="Created by Ng Chee Qi",
            font=("Arial", 11, "bold"),
            foreground="#1971c2"
        )
        credit_label.grid(row=1, column=0, columnspan=2, pady=(0, 15))

        # Subtitle
        subtitle_label = ttk.Label(
            main_frame,
            text="Enter gene features to predict heat-stress response",
            font=("Arial", 10)
        )
        subtitle_label.grid(row=2, column=0, columnspan=2, pady=(0, 20))

        # Feature input section
        input_frame = ttk.LabelFrame(main_frame, text="Gene Features", padding="15")
        input_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))

        # Feature descriptions
        feature_info = {
            'log2FC': ('Log2 Fold Change:', 'e.g., 2.3 (range: -10 to 10)'),
            'neg_log10_pvalue': ('Negative Log10 P-value:', 'e.g., 5.1 (range: 0 to 50)'),
            'baseMean': ('Base Mean Expression:', 'e.g., 600 (range: 0 to 100000)'),
            'gene_length': ('Gene Length (bp):', 'e.g., 1500 (range: 100 to 20000)'),
            'GC_content': ('GC Content:', 'e.g., 0.42 (range: 0 to 1)')
        }

        # Create input fields
        self.entries = {}
        row = 0

        for feature, (label_text, hint) in feature_info.items():
            # Label
            label = ttk.Label(input_frame, text=label_text, font=("Arial", 10, "bold"))
            label.grid(row=row, column=0, sticky=tk.W, pady=5)

            # Entry
            entry = ttk.Entry(input_frame, width=30, font=("Arial", 10))
            entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
            self.entries[feature] = entry

            # Hint
            hint_label = ttk.Label(input_frame, text=hint, font=("Arial", 8), foreground="gray")
            hint_label.grid(row=row, column=2, sticky=tk.W, pady=5, padx=(10, 0))

            row += 1

        # Example button
        example_btn = ttk.Button(
            input_frame,
            text="Load Example Gene",
            command=self.load_example
        )
        example_btn.grid(row=row, column=0, columnspan=3, pady=(10, 0))

        # Predict button
        predict_btn = ttk.Button(
            main_frame,
            text="Predict",
            command=self.predict,
            style="Accent.TButton"
        )
        predict_btn.grid(row=4, column=0, columnspan=2, pady=(0, 20), ipadx=20, ipady=5)

        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="15")
        results_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Results text area (scrollable)
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            width=70,
            height=20,
            font=("Courier", 10),
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure text tags for color coding
        self.results_text.tag_configure("responsive", foreground=self.color_responsive, font=("Courier", 10, "bold"))
        self.results_text.tag_configure("non_responsive", foreground=self.color_non_responsive, font=("Courier", 10, "bold"))
        self.results_text.tag_configure("ensemble", foreground="#1971c2", font=("Courier", 12, "bold"))
        self.results_text.tag_configure("header", font=("Courier", 10, "bold"))

        # Clear button
        clear_btn = ttk.Button(
            main_frame,
            text="Clear All",
            command=self.clear_all
        )
        clear_btn.grid(row=6, column=0, columnspan=2, pady=(10, 0))

    def load_example(self):
        """Load example gene data into input fields."""
        example_data = {
            'log2FC': '2.3',
            'neg_log10_pvalue': '5.1',
            'baseMean': '600',
            'gene_length': '1500',
            'GC_content': '0.42'
        }

        for feature, value in example_data.items():
            self.entries[feature].delete(0, tk.END)
            self.entries[feature].insert(0, value)

    def clear_all(self):
        """Clear all input fields and results."""
        for entry in self.entries.values():
            entry.delete(0, tk.END)

        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)

    def validate_inputs(self):
        """
        Validate all input fields.

        Returns:
            Tuple of (is_valid, features_dict or error_message)
        """
        features = {}

        # Validation ranges
        validations = {
            'log2FC': (-10, 10, "log2 fold change should be between -10 and 10"),
            'neg_log10_pvalue': (0, 50, "negative log10 p-value should be between 0 and 50"),
            'baseMean': (0, 100000, "baseMean should be between 0 and 100,000"),
            'gene_length': (100, 20000, "gene_length should be between 100 and 20,000 bp"),
            'GC_content': (0, 1, "GC_content should be between 0 and 1")
        }

        for feature in self.feature_columns:
            value_str = self.entries[feature].get().strip()

            # Check if empty
            if not value_str:
                return False, f"Please enter a value for {feature}"

            # Try to convert to float
            try:
                if feature == 'gene_length':
                    value = int(value_str)
                else:
                    value = float(value_str)
            except ValueError:
                return False, f"Invalid value for {feature}: '{value_str}'\nMust be a number."

            # Check range
            min_val, max_val, msg = validations[feature]
            if not (min_val <= value <= max_val):
                return False, f"Invalid {feature}={value}\n{msg}"

            features[feature] = value

        return True, features

    def predict(self):
        """Make prediction based on input features."""
        # Validate inputs
        is_valid, result = self.validate_inputs()
        if not is_valid:
            messagebox.showerror("Validation Error", result)
            return

        features = result

        try:
            # Create DataFrame with correct column order
            features_df = pd.DataFrame([features])[self.feature_columns]

            # Make predictions
            # RandomForest (no scaling)
            rf_pred = self.rf_model.predict(features_df)[0]
            rf_prob = self.rf_model.predict_proba(features_df)[0, 1]

            # LogisticRegression (with scaling)
            features_scaled_lr = self.scaler_lr.transform(features_df)
            lr_pred = self.lr_model.predict(features_scaled_lr)[0]
            lr_prob = self.lr_model.predict_proba(features_scaled_lr)[0, 1]

            # SVM (with scaling)
            features_scaled_svm = self.scaler_svm.transform(features_df)
            svm_pred = self.svm_model.predict(features_scaled_svm)[0]
            svm_prob = self.svm_model.predict_proba(features_scaled_svm)[0, 1]

            # Ensemble prediction
            ensemble_prob = (rf_prob + lr_prob + svm_prob) / 3
            ensemble_pred = 1 if ensemble_prob > 0.5 else 0

            # Display results
            self.display_results(features, {
                'RandomForest': {'prediction': rf_pred, 'probability': rf_prob},
                'LogisticRegression': {'prediction': lr_pred, 'probability': lr_prob},
                'SVM': {'prediction': svm_pred, 'probability': svm_prob},
                'Ensemble': {'prediction': ensemble_pred, 'probability': ensemble_prob}
            })

        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred:\n\n{str(e)}")

    def display_results(self, features, predictions):
        """
        Display prediction results in the text area.

        Args:
            features: Dictionary of input features
            predictions: Dictionary of model predictions
        """
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        # Input features
        self.results_text.insert(tk.END, "=" * 70 + "\n", "header")
        self.results_text.insert(tk.END, "INPUT FEATURES\n", "header")
        self.results_text.insert(tk.END, "=" * 70 + "\n", "header")
        for feature, value in features.items():
            self.results_text.insert(tk.END, f"{feature:20s}: {value}\n")

        self.results_text.insert(tk.END, "\n")

        # Model predictions
        self.results_text.insert(tk.END, "=" * 70 + "\n", "header")
        self.results_text.insert(tk.END, "PREDICTION RESULTS\n", "header")
        self.results_text.insert(tk.END, "=" * 70 + "\n", "header")
        self.results_text.insert(tk.END, "\n")

        for model_name, result in predictions.items():
            pred = result['prediction']
            prob = result['probability']
            label = "RESPONSIVE" if pred == 1 else "Non-responsive"
            confidence = prob if pred == 1 else (1 - prob)

            if model_name == 'Ensemble':
                self.results_text.insert(tk.END, "-" * 70 + "\n")
                self.results_text.insert(tk.END, f"🔥 {model_name.upper()} PREDICTION:\n", "ensemble")
            else:
                self.results_text.insert(tk.END, f"{model_name}:\n", "header")

            # Color-coded prediction
            tag = "responsive" if pred == 1 else "non_responsive"
            self.results_text.insert(tk.END, f"  Prediction: ", "header")
            self.results_text.insert(tk.END, f"{label}\n", tag)

            self.results_text.insert(tk.END, f"  Probability (responsive): {prob:.4f}\n")
            self.results_text.insert(tk.END, f"  Confidence: {confidence:.2%}\n")
            self.results_text.insert(tk.END, "\n")

        self.results_text.insert(tk.END, "=" * 70 + "\n", "header")

        # Interpretation
        ensemble_pred = predictions['Ensemble']['prediction']
        ensemble_prob = predictions['Ensemble']['probability']

        self.results_text.insert(tk.END, "\nINTERPRETATION:\n", "header")
        if ensemble_pred == 1:
            self.results_text.insert(
                tk.END,
                f"This gene is predicted to be HEAT-STRESS RESPONSIVE with "
                f"{ensemble_prob:.1%} confidence.\n",
                "responsive"
            )
            self.results_text.insert(
                tk.END,
                "\nThis gene likely shows significant differential expression under "
                "heat stress conditions and may play a role in the plant's heat "
                "stress response mechanism.\n"
            )
        else:
            self.results_text.insert(
                tk.END,
                f"This gene is predicted to be NON-RESPONSIVE with "
                f"{(1-ensemble_prob):.1%} confidence.\n",
                "non_responsive"
            )
            self.results_text.insert(
                tk.END,
                "\nThis gene likely does not show significant differential expression "
                "under heat stress conditions based on the provided features.\n"
            )

        self.results_text.config(state=tk.DISABLED)


def main():
    """Main entry point for GUI application."""
    # Determine models directory
    # Check if running from heat_stress_predictor directory
    if Path('models').exists():
        models_dir = 'models'
    elif Path('../models').exists():
        models_dir = '../models'
    else:
        # Try to find models in same directory as script
        script_dir = Path(__file__).parent
        models_dir = script_dir / 'models'

    # Create main window
    root = tk.Tk()

    # Create application
    app = GenePredictorGUI(root, models_dir=str(models_dir))

    # Start GUI event loop
    root.mainloop()


if __name__ == '__main__':
    main()
