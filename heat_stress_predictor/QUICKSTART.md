# 🚀 Quick Start Guide

Get up and running with the Heat Stress Gene Predictor in 5 minutes!

---

## ⚡ Fast Setup (3 Steps)

### Step 1: Generate Models

First, run the machine learning notebook to train models:

```bash
# From the parent directory (land-plants-prediction)
jupyter notebook ML_HEAT_STRESS_NOTEBOOK.ipynb
```

In Jupyter:
1. Click **Cell → Run All**
2. Wait ~3 minutes for training to complete
3. Verify these files are created:
   - ✅ `rf_model.pkl`
   - ✅ `lr_model.pkl`
   - ✅ `svm_model.pkl`
   - ✅ `scaler_lr.pkl`
   - ✅ `scaler_svm.pkl`
   - ✅ `feature_columns.json`

### Step 2: Copy Models

```bash
cd heat_stress_predictor
python setup_models.py
```

You should see:
```
✓ All model files successfully copied!
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🎮 Test the System

### Test 1: CLI Prediction

```bash
python predict_gene.py --log2FC 2.3 --neg_log10_pvalue 5.1 --baseMean 600 --gene_length 1500 --GC_content 0.42
```

**Expected:** You should see prediction results from all 3 models + ensemble.

### Test 2: GUI

```bash
python gene_predictor_gui.py
```

**Expected:** Desktop window opens with input fields.

### Test 3: Dashboard

```bash
streamlit run app.py
```

**Expected:** Browser opens to `http://localhost:8501` with interactive dashboard.

---

## 📝 Example Workflow

### Single Gene Prediction (CLI)

```bash
# Responsive gene example
python predict_gene.py \
    --log2FC 3.5 \
    --neg_log10_pvalue 10.2 \
    --baseMean 2000 \
    --gene_length 1800 \
    --GC_content 0.48
```

### Batch Prediction

1. Create `my_genes.csv`:
```csv
gene_id,log2FC,neg_log10_pvalue,baseMean,gene_length,GC_content
HSP70,3.2,12.5,2500,1950,0.45
Unknown,0.6,1.8,600,1200,0.52
```

2. Run prediction:
```bash
python predict_gene.py --input my_genes.csv --output results.csv
```

3. View results:
```bash
head results.csv  # Linux/Mac
type results.csv  # Windows
```

---

## 🎯 Common Use Cases

### Use Case 1: Screen Gene List

**Scenario:** You have 100 candidate genes from RNA-seq analysis.

**Solution:**
1. Export genes to CSV with 5 required features
2. Run: `python predict_gene.py --input genes.csv --output predictions.csv`
3. Filter results: genes with `Ensemble_probability > 0.8`
4. Validate top candidates with qPCR

### Use Case 2: Interactive Exploration

**Scenario:** Exploring feature impacts on predictions.

**Solution:**
1. Launch dashboard: `streamlit run app.py`
2. Use sliders to adjust features in real-time
3. Observe how probability changes
4. Check feature importance plots

### Use Case 3: High-Throughput Pipeline

**Scenario:** Integrate into automated analysis pipeline.

**Solution:**
```python
# Python script
from predict_gene import GenePredictor

predictor = GenePredictor(models_dir='models')
results = predictor.predict_batch('input.csv', 'output.csv', verbose=False)
responsive_genes = results[results['Ensemble_prediction'] == 1]
```

---

## 🛠️ Troubleshooting Quick Fixes

### Issue: "Models directory not found"

```bash
# Check you're in the right directory
pwd  # Should show heat_stress_predictor

# Check models exist
ls models/  # Should show 6 files
```

**Fix:** Run `python setup_models.py`

### Issue: "Module not found"

```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Issue: Streamlit won't start

```bash
# Try different port
streamlit run app.py --server.port 8502
```

---

## 📊 Feature Value Cheat Sheet

Quick reference for input values:

| Feature | Low | Medium | High |
|---------|-----|--------|------|
| log2FC | -2.0 | 0.0 | +3.0 |
| neg_log10_pvalue | 0.5 | 2.0 | 10.0 |
| baseMean | 100 | 1000 | 5000 |
| gene_length | 500 | 1500 | 5000 |
| GC_content | 0.35 | 0.45 | 0.60 |

**Typical Responsive Gene:**
- log2FC: 2.0 to 4.0
- neg_log10_pvalue: 5.0 to 20.0
- Others: variable

---

## 🎓 Next Steps

1. **Read Full Documentation**: [README.md](README.md)
2. **Explore Dashboard**: Check all 4 tabs in Streamlit app
3. **Try Examples**: Use provided example genes
4. **Customize**: Modify thresholds and settings

---

## 💡 Pro Tips

✨ **CLI Tip:** Use `\` for line continuation in long commands
✨ **GUI Tip:** Click "Load Example" button for quick testing
✨ **Dashboard Tip:** Download example CSV to see exact format
✨ **Batch Tip:** Sort output by probability to find top candidates
✨ **Performance Tip:** Ensemble prediction is most reliable

---

## ❓ Need Help?

- **Documentation**: [README.md](README.md)
- **Model Info**: Check "Model Info" tab in dashboard
- **Examples**: See "Documentation" tab in dashboard
- **Issues**: Check [Troubleshooting section](README.md#troubleshooting) in README

---

**That's it! You're ready to predict heat-stress genes! 🔥**
