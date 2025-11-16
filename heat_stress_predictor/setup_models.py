#!/usr/bin/env python3
"""
Helper script to copy model files from parent directory to models/ folder.

Run this after executing ML_HEAT_STRESS_NOTEBOOK.ipynb to set up the prediction system.

Usage:
    python setup_models.py
"""

import shutil
from pathlib import Path
import sys


def setup_models():
    """Copy model files to models directory."""
    # Define paths
    parent_dir = Path(__file__).parent.parent
    models_dir = Path(__file__).parent / 'models'

    # Create models directory if it doesn't exist
    models_dir.mkdir(exist_ok=True)

    # List of required files
    required_files = [
        'rf_model.pkl',
        'lr_model.pkl',
        'svm_model.pkl',
        'scaler_lr.pkl',
        'scaler_svm.pkl',
        'feature_columns.json'
    ]

    print("Setting up model files...")
    print(f"Source directory: {parent_dir}")
    print(f"Destination directory: {models_dir}\n")

    missing_files = []
    copied_files = []

    for filename in required_files:
        source = parent_dir / filename
        destination = models_dir / filename

        if source.exists():
            shutil.copy2(source, destination)
            copied_files.append(filename)
            print(f"[OK] Copied: {filename}")
        else:
            missing_files.append(filename)
            print(f"[MISSING] {filename}")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Copied: {len(copied_files)}/{len(required_files)} files")

    if missing_files:
        print(f"\n[WARNING] Missing files: {missing_files}")
        print(f"\nPlease run ML_HEAT_STRESS_NOTEBOOK.ipynb to generate these files.")
        print(f"\nSteps:")
        print(f"  1. Open ML_HEAT_STRESS_NOTEBOOK.ipynb in Jupyter")
        print(f"  2. Run all cells (Cell -> Run All)")
        print(f"  3. Wait for all models to train and save")
        print(f"  4. Run this script again: python setup_models.py")
        return False
    else:
        print(f"\n[SUCCESS] All model files successfully copied!")
        print(f"\nYou can now use the prediction system:")
        print(f"  - CLI: python predict_gene.py --help")
        print(f"  - GUI: python gene_predictor_gui.py")
        print(f"  - Dashboard: streamlit run app.py")
        return True


if __name__ == '__main__':
    success = setup_models()
    sys.exit(0 if success else 1)
