# Heat Stress Gene Predictor

A machine learning system for predicting heat-stress-responsive genes in land plants using gene expression and genomic features.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This project implements three machine learning classifiers (RandomForest, Logistic Regression, SVM) to identify heat-stress-responsive genes based on:
- Differential expression (log2 fold change)
- Statistical significance (p-value)
- Expression level (baseMean)
- Genomic features (gene length, GC content)

### Model Performance

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| RandomForest | 100.00% | 0.9983 |
| LogisticRegression | 99.33% | 0.9952 |
| SVM | 99.17% | 0.9967 |
| **Ensemble** | **99.89%** | **0.9975** |

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Generate models (first time only)
python generate_models.py

# Copy models to prediction system
cd heat_stress_predictor
python setup_models.py
```

### Usage

**Command Line:**
```bash
cd heat_stress_predictor
python predict_gene.py --log2FC 2.3 --neg_log10_pvalue 5.1 --baseMean 600 --gene_length 1500 --GC_content 0.42
```

**GUI Application:**
```bash
cd heat_stress_predictor
python gene_predictor_gui.py
# or double-click launch_gui.bat on Windows
```

**Web Dashboard:**
```bash
cd heat_stress_predictor
streamlit run app.py
```

---

## Project Structure

```
land-plants-prediction/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── generate_models.py                 # Model training script
│
├── Notebooks
├── ML_HEAT_STRESS_NOTEBOOK.ipynb     # Full ML workflow
├── heat_stress_ml_project.ipynb      # Original research notebook
├── heat_stress_data.csv              # Training dataset (500 genes)
│
├── Models (generated)
├── rf_model.pkl                      # RandomForest classifier
├── lr_model.pkl                      # Logistic Regression
├── svm_model.pkl                     # SVM classifier
├── scaler_lr.pkl                     # Feature scaler (LR)
├── scaler_svm.pkl                    # Feature scaler (SVM)
├── feature_columns.json              # Feature metadata
│
├── Visualizations (from notebooks)
├── volcano_plot.png                  # Differential expression
├── correlation_heatmap.png           # Feature correlations
├── feature_importance.png            # RF importance scores
├── roc_curve.png                     # Model comparison
├── confusion_matrix.png              # Classification errors
├── heatmap.png                       # Gene expression patterns
└── gene_network.png                  # Simulated PPI network
│
└── heat_stress_predictor/            # Prediction system
    ├── models/                       # Model artifacts
    ├── predict_gene.py              # CLI tool
    ├── gene_predictor_gui.py        # Desktop GUI
    ├── app.py                       # Streamlit dashboard
    ├── launch_gui.bat               # Windows launcher
    ├── README.md                    # Detailed documentation
    └── QUICKSTART.md                # 5-minute guide
```

---

## Features

### Input Features

| Feature | Description | Range |
|---------|-------------|-------|
| `log2FC` | Log2 fold change (expression) | -10 to 10 |
| `neg_log10_pvalue` | -log10(p-value) significance | 0 to 50 |
| `baseMean` | Average expression level | 0 to 100,000 |
| `gene_length` | Gene length in base pairs | 100 to 20,000 |
| `GC_content` | GC nucleotide proportion | 0 to 1 |

### Label Definition

A gene is classified as **heat-stress-responsive** if:
```
(log2FC > 1.0) AND (p_value < 0.05)
```

This indicates significant upregulation (>2-fold change) under heat stress conditions.

---

## Prediction System

The `heat_stress_predictor/` directory contains three interfaces:

### 1. Command Line Interface

**Single gene prediction:**
```bash
python predict_gene.py \
    --log2FC 2.3 \
    --neg_log10_pvalue 5.1 \
    --baseMean 600 \
    --gene_length 1500 \
    --GC_content 0.42
```

**Batch prediction:**
```bash
python predict_gene.py --input genes.csv --output predictions.csv
```

**CSV format:**
```csv
gene_id,log2FC,neg_log10_pvalue,baseMean,gene_length,GC_content
Gene1,2.3,5.1,600,1500,0.42
Gene2,0.5,1.8,1200,2000,0.55
```

### 2. Desktop GUI

Launch with:
```bash
python gene_predictor_gui.py
# or double-click launch_gui.bat on Windows
```

Features:
- Input fields for all 5 features
- Load example button
- Color-coded predictions
- Confidence scores for all models

### 3. Web Dashboard (Streamlit)

Launch with:
```bash
streamlit run app.py
```

Includes:
- **Single Gene Prediction** - Interactive sliders
- **Batch Prediction** - CSV upload and download
- **Model Info** - Feature importance visualizations
- **Documentation** - Complete usage guide

---

## Research Notebooks

### ML_HEAT_STRESS_NOTEBOOK.ipynb

Complete machine learning workflow:
1. Synthetic data generation (2,000 genes)
2. Train/test split (70/30)
3. Model training (RF, LR, SVM)
4. Evaluation and visualization
5. Feature importance analysis
6. Model saving

### heat_stress_ml_project.ipynb

Original research notebook with:
- RNA-seq analysis workflow
- Differential expression analysis
- Biological interpretation
- Gene network visualization
- Functional enrichment guidance

---

## Model Training

To retrain models from scratch:

```bash
# Option 1: Use Python script
python generate_models.py

# Option 2: Use Jupyter notebook
jupyter notebook ML_HEAT_STRESS_NOTEBOOK.ipynb
# Run all cells

# Then copy models to prediction system
cd heat_stress_predictor
python setup_models.py
```

---

## Requirements

- Python 3.11+
- numpy >= 1.26.4
- pandas >= 2.3.3
- scikit-learn >= 1.7.2
- matplotlib >= 3.10.7
- seaborn >= 0.13.2
- streamlit >= 1.32.0
- plotly >= 5.19.0

Install all:
```bash
pip install -r requirements.txt
```

---

## Use Cases

- **Gene Discovery**: Identify candidate genes for experimental validation
- **Crop Breeding**: Prioritize genes for heat tolerance engineering
- **Comparative Genomics**: Compare stress responses across species
- **Education**: Teaching bioinformatics and machine learning

---

## Biological Context

Heat-stress-responsive genes typically include:
- **HSP70/HSP90** - Heat shock proteins (molecular chaperones)
- **HSF** - Heat shock transcription factors
- **DREB** - Dehydration-responsive element binding proteins
- **LEA** - Late embryogenesis abundant proteins

These genes are crucial for plant survival under elevated temperatures and are conserved across land plant evolution.

---

## Example Results

```
>>> ENSEMBLE PREDICTION <<<
  Prediction: RESPONSIVE
  Probability (responsive): 0.9989
  Confidence: 99.89%
```

---

## Troubleshooting

**Models not found:**
```bash
# Generate models first
python generate_models.py
cd heat_stress_predictor
python setup_models.py
```

**Import errors:**
```bash
pip install -r requirements.txt
```

**GUI won't launch:**
- Windows: Double-click `launch_gui.bat`
- Ensure Tkinter is installed (included with Python)

**Streamlit port in use:**
```bash
streamlit run app.py --server.port 8502
```

---

## Dataset

The training dataset (`heat_stress_data.csv`) contains:
- 500 simulated genes
- RNA-seq expression features
- Genomic properties
- Binary labels (responsive/non-responsive)

Responsive genes show:
- log2FC > 1 (>2-fold upregulation)
- p < 0.05 (statistical significance)

---

## Contributing

Contributions are welcome! Areas for improvement:
- Real RNA-seq data integration
- Additional ML algorithms
- Feature engineering
- Cross-species validation

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

Developed for plant genomics research and bioinformatics education.

Built with:
- scikit-learn (Machine Learning)
- Streamlit (Web Dashboard)
- Tkinter (Desktop GUI)
- Plotly (Interactive Visualizations)

---

## Version

**Version:** 1.0.0
**Last Updated:** November 2025
**Status:** Production Ready

---

For detailed documentation, see:
- [heat_stress_predictor/README.md](heat_stress_predictor/README.md) - Complete prediction system guide
- [heat_stress_predictor/QUICKSTART.md](heat_stress_predictor/QUICKSTART.md) - 5-minute setup guide
