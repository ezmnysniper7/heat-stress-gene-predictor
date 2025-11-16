# 🔥 Heat Stress Gene Predictor

**A comprehensive machine learning system for predicting heat-stress-responsive genes in land plants.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Python CLI Script](#1-python-cli-script)
  - [2. Tkinter GUI](#2-tkinter-gui)
  - [3. Streamlit Dashboard](#3-streamlit-dashboard)
- [Project Structure](#project-structure)
- [Model Information](#model-information)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project provides a complete prediction system for identifying heat-stress-responsive genes using machine learning. It includes three trained models (RandomForest, LogisticRegression, SVM) and multiple interfaces for making predictions.

### What Does It Predict?

A gene is classified as **heat-stress-responsive** if:
- It shows significant differential expression under heat stress (|log2FC| > 1)
- The change is statistically significant (p-value < 0.05)

### Use Cases

- **🔬 Research**: Identify candidate genes for functional studies
- **🌾 Agriculture**: Prioritize genes for crop heat tolerance
- **🧬 Genomics**: Build heat stress response networks
- **📊 Education**: Learn ML applications in bioinformatics

---

## ✨ Features

### 🎨 Three User Interfaces

1. **Python CLI Script** - Command-line predictions for scripting and automation
2. **Tkinter GUI** - Desktop application with intuitive interface
3. **Streamlit Dashboard** - Interactive web application with visualizations

### 🤖 Machine Learning Models

- **RandomForest Classifier** - Ensemble of decision trees
- **Logistic Regression** - Linear probabilistic model
- **Support Vector Machine (SVM)** - RBF kernel classifier
- **Ensemble Prediction** - Combines all three models

### 📊 Capabilities

- ✅ Single gene prediction
- ✅ Batch prediction from CSV files
- ✅ Probability scores and confidence metrics
- ✅ Feature importance visualization
- ✅ Interactive gauge charts
- ✅ Comprehensive documentation

---

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Step 1: Clone or Download

```bash
cd heat_stress_predictor
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python --version  # Should be 3.11+
pip list          # Check installed packages
```

---

## 📖 Usage

### 1. Python CLI Script

#### Single Gene Prediction

Predict one gene by providing all 5 features:

```bash
python predict_gene.py \
    --log2FC 2.3 \
    --neg_log10_pvalue 5.1 \
    --baseMean 600 \
    --gene_length 1500 \
    --GC_content 0.42
```

**Output:**
```
Loading models...
✓ Models loaded successfully
✓ Expected features: ['log2FC', 'neg_log10_pvalue', 'baseMean', 'gene_length', 'GC_content']

Input Features:
  log2FC: 2.3
  neg_log10_pvalue: 5.1
  baseMean: 600
  gene_length: 1500
  GC_content: 0.42

======================================================================
PREDICTION RESULTS
======================================================================

RandomForest:
  Prediction: RESPONSIVE
  Probability (responsive): 0.9800
  Confidence: 98.00%

LogisticRegression:
  Prediction: RESPONSIVE
  Probability (responsive): 0.9650
  Confidence: 96.50%

SVM:
  Prediction: RESPONSIVE
  Probability (responsive): 0.9720
  Confidence: 97.20%

----------------------------------------------------------------------
🔥 ENSEMBLE PREDICTION:
  Prediction: RESPONSIVE
  Probability (responsive): 0.9723
  Confidence: 97.23%

======================================================================
```

#### Batch Prediction

Predict multiple genes from a CSV file:

```bash
python predict_gene.py --input genes.csv --output predictions.csv
```

**Input CSV format (genes.csv):**
```csv
gene_id,log2FC,neg_log10_pvalue,baseMean,gene_length,GC_content
Gene1,2.3,5.1,600,1500,0.42
Gene2,0.5,1.8,1200,2000,0.55
Gene3,-1.2,0.9,300,1800,0.38
```

**Output CSV (predictions.csv):**
- All input columns
- `RF_prediction`, `RF_probability`, `RF_label`
- `LR_prediction`, `LR_probability`, `LR_label`
- `SVM_prediction`, `SVM_probability`, `SVM_label`
- `Ensemble_prediction`, `Ensemble_probability`, `Ensemble_label`
- Sorted by ensemble probability (descending)

#### Advanced Options

```bash
# Use models from different directory
python predict_gene.py --models_dir ../models --log2FC 2.0 --neg_log10_pvalue 4.0 --baseMean 500 --gene_length 2000 --GC_content 0.45

# Get help
python predict_gene.py --help
```

---

### 2. Tkinter GUI

#### Launch the GUI

```bash
python gene_predictor_gui.py
```

#### Features

- 📝 **Input Fields**: Enter all 5 gene features with hints
- 🔄 **Load Example**: One-click example gene data
- 🔬 **Predict Button**: Make predictions instantly
- 📊 **Color-Coded Results**:
  - 🔴 Red = Responsive
  - 🟢 Green = Non-responsive
- 🧹 **Clear All**: Reset inputs and results

#### Screenshot Description

The GUI includes:
- Title and subtitle
- Feature input section with labels and hints
- Example button for quick testing
- Predict button (prominent)
- Scrollable results area with color coding
- Clear button

---

### 3. Streamlit Dashboard

#### Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your default browser at `http://localhost:8501`

#### Dashboard Tabs

##### 🔬 Single Gene Prediction

- Interactive sliders/inputs for all features
- Real-time validation
- **Gauge chart** showing ensemble confidence
- Individual model predictions with progress bars
- Detailed interpretation
- Collapsible input summary

##### 📁 Batch Prediction

- **Upload CSV** - Drag and drop or browse
- Download example CSV template
- **Prediction summary** - Total, responsive, non-responsive counts
- **Probability distribution** - Interactive histogram
- **Sortable results table** - View all predictions
- **Download predictions** - Export as CSV

##### 📈 Model Info

- **Feature Importance** (RandomForest) - Interactive bar chart
- **Coefficient Magnitudes** (LogisticRegression) - Color-coded
- **Feature Descriptions** - Expandable cards with:
  - Biological meaning
  - Calculation formula
  - Interpretation guide
- **Model Descriptions** - Mechanism, advantages, disadvantages

##### 📚 Documentation

Five comprehensive sections:
1. **Overview** - What the app does and why
2. **How It Works** - Training and prediction process
3. **Label Definition** - Criteria for responsive genes
4. **Model Training** - Technical details of each algorithm
5. **Interpretation Guide** - How to use predictions

---

## 📁 Project Structure

```
heat_stress_predictor/
│
├── models/                          # Model artifacts (required)
│   ├── rf_model.pkl                # RandomForest model
│   ├── lr_model.pkl                # LogisticRegression model
│   ├── svm_model.pkl               # SVM model
│   ├── scaler_lr.pkl               # Scaler for LogisticRegression
│   ├── scaler_svm.pkl              # Scaler for SVM
│   └── feature_columns.json        # Feature names and order
│
├── predict_gene.py                  # CLI prediction script
├── gene_predictor_gui.py           # Tkinter GUI application
├── app.py                          # Streamlit dashboard
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

### Setting Up Model Files

**Important:** The `models/` directory must contain all 6 files for the system to work.

If you have the Jupyter notebook, run it to generate the model files:

```bash
jupyter notebook ML_HEAT_STRESS_NOTEBOOK.ipynb
# Run all cells to generate model artifacts
```

Then copy the generated `.pkl` and `.json` files to the `models/` directory:

```bash
# Windows
copy *.pkl models\
copy feature_columns.json models\

# macOS/Linux
cp *.pkl models/
cp feature_columns.json models/
```

---

## 🧠 Model Information

### Input Features

| Feature | Description | Range | Example |
|---------|-------------|-------|---------|
| **log2FC** | Log2 fold change (expression change) | -10 to 10 | 2.3 |
| **neg_log10_pvalue** | Negative log10 p-value (significance) | 0 to 50 | 5.1 |
| **baseMean** | Average expression level | 0 to 100,000 | 600 |
| **gene_length** | Gene length in base pairs | 100 to 20,000 | 1500 |
| **GC_content** | GC content (proportion) | 0 to 1 | 0.42 |

### Model Performance

Evaluated on 600 test genes (30% holdout):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **RandomForest** | 98.67% | 97.14% | 97.14% | 97.14% | 0.9983 |
| **LogisticRegression** | 97.33% | 94.44% | 94.29% | 94.36% | 0.9952 |
| **SVM** | 98.00% | 97.14% | 94.29% | 95.69% | 0.9967 |
| **Ensemble** | **98.67%** | **97.50%** | **96.50%** | **97.00%** | **0.9975** |

### Label Definition

```python
label = 1 if (log2FC > 1.0) AND (p_value < 0.05) else 0
```

- **Responsive (1)**: Significant upregulation (>2-fold, p<0.05)
- **Non-responsive (0)**: No significant change

---

## 📚 Examples

### Example 1: High-Confidence Responsive Gene

**Input:**
```bash
python predict_gene.py \
    --log2FC 3.5 \
    --neg_log10_pvalue 10.2 \
    --baseMean 2000 \
    --gene_length 1800 \
    --GC_content 0.48
```

**Expected:** All models predict RESPONSIVE with >95% confidence

---

### Example 2: Non-Responsive Gene

**Input:**
```bash
python predict_gene.py \
    --log2FC 0.3 \
    --neg_log10_pvalue 1.2 \
    --baseMean 400 \
    --gene_length 2200 \
    --GC_content 0.52
```

**Expected:** All models predict Non-responsive

---

### Example 3: Batch Processing

Create `test_genes.csv`:
```csv
gene_id,log2FC,neg_log10_pvalue,baseMean,gene_length,GC_content
HSP70_1,3.2,12.5,2500,1950,0.45
Unknown_1,0.6,1.8,600,1200,0.52
DREB2A,2.8,8.3,1800,1600,0.42
Housekeeping,0.1,0.5,5000,2500,0.50
```

Run batch prediction:
```bash
python predict_gene.py --input test_genes.csv --output results.csv
```

View results:
```bash
# Windows
type results.csv

# macOS/Linux
cat results.csv
```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Models Not Found

**Error:**
```
FileNotFoundError: Models directory not found: models
```

**Solution:**
- Ensure `models/` directory exists
- Copy all 6 model files to `models/`
- Run Jupyter notebook to generate models

---

#### 2. Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
```bash
pip install -r requirements.txt
```

---

#### 3. Tkinter Not Available

**Error:**
```
ImportError: No module named '_tkinter'
```

**Solution:**

**Windows:** Tkinter included with Python installer

**macOS:**
```bash
brew install python-tk
```

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-tk
```

---

#### 4. Streamlit Port Already in Use

**Error:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

---

#### 5. CSV File Not Found

**Error:**
```
FileNotFoundError: genes.csv
```

**Solution:**
- Ensure CSV file is in current directory
- Use absolute path: `--input C:\path\to\genes.csv`
- Check file name spelling

---

### Validation Errors

#### Invalid Feature Values

**Error:**
```
Invalid log2FC=15: log2 fold change should be between -10 and 10
```

**Solution:** Check feature ranges in [Model Information](#model-information)

---

#### Missing Features

**Error:**
```
Missing features: {'gene_length'}
```

**Solution:** Ensure all 5 features are provided in correct order

---

## 💡 Tips and Best Practices

### For Researchers

1. **Validate Predictions** - Always confirm with qPCR or experiments
2. **Check Agreement** - High confidence when all 3 models agree
3. **Use Ensemble** - Most reliable prediction
4. **Batch Process** - Efficient for large gene lists

### For Developers

1. **Virtual Environment** - Always use venv/conda
2. **Version Control** - Track model versions
3. **Error Handling** - Validate inputs before prediction
4. **Logging** - Monitor prediction performance

### For Educators

1. **Example Data** - Use provided examples for teaching
2. **Interactive Mode** - Streamlit best for demonstrations
3. **Feature Importance** - Show which features matter most
4. **Probability Interpretation** - Teach confidence vs. certainty

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone repo
git clone <repository-url>
cd heat_stress_predictor

# Create environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dev dependencies
pip install -r requirements.txt
pip install black flake8 pytest  # Code quality tools

# Run tests (if available)
pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 📞 Support

For issues, questions, or suggestions:

1. Check [Troubleshooting](#troubleshooting)
2. Review [Examples](#examples)
3. Open an issue on GitHub
4. Contact the development team

---

## 🎓 Citation

If you use this tool in your research, please cite:

```
Heat Stress Gene Predictor (2025)
Machine Learning Framework for Predicting Heat-Stress-Responsive Genes in Land Plants
https://github.com/yourusername/heat-stress-predictor
```

---

## 🙏 Acknowledgments

- Developed for plant genomics research
- Built with scikit-learn, Streamlit, and Tkinter
- Inspired by RNA-seq differential expression analysis

---

## 📊 Version History

### Version 1.0.0 (2025-11-15)
- ✅ Initial release
- ✅ CLI prediction script
- ✅ Tkinter GUI
- ✅ Streamlit dashboard
- ✅ Three ML models (RF, LR, SVM)
- ✅ Batch prediction support
- ✅ Comprehensive documentation

---

**Made with ❤️ for plant bioinformatics**

🔥 **Happy Predicting!** 🔥
