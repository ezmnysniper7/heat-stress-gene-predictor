MODEL FILES DIRECTORY
=====================

This directory should contain 6 model artifact files:

1. rf_model.pkl           - RandomForest classifier
2. lr_model.pkl           - LogisticRegression classifier
3. svm_model.pkl          - SVM classifier
4. scaler_lr.pkl          - StandardScaler for LogisticRegression
5. scaler_svm.pkl         - StandardScaler for SVM
6. feature_columns.json   - Feature names and order

HOW TO GENERATE THESE FILES:
----------------------------

Step 1: Run the ML notebook in the parent directory
   cd ..
   jupyter notebook ML_HEAT_STRESS_NOTEBOOK.ipynb

Step 2: In Jupyter, run all cells (Cell → Run All)
   This will train the models and save the files

Step 3: Copy the files to this directory
   cd heat_stress_predictor
   python setup_models.py

Once these files are in place, you can use:
- python predict_gene.py (CLI)
- python gene_predictor_gui.py (GUI)
- streamlit run app.py (Dashboard)

For more information, see:
- QUICKSTART.md - Quick setup guide
- README.md - Full documentation
