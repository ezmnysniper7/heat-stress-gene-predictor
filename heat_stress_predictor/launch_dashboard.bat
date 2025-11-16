@echo off
echo ========================================
echo Heat Stress Gene Predictor - Dashboard
echo ========================================
echo.
echo Launching Streamlit dashboard...
echo The dashboard will open in your browser at http://localhost:8501
echo.

python -m streamlit run app.py

echo.
echo Dashboard closed.
pause
