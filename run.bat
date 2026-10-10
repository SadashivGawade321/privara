@echo off
echo ============================================================
echo   Privara - Privacy-Preserving Federated Learning Simulator
echo   TECHNOVA 2026 Edition
echo ============================================================
echo.
echo Starting Streamlit dashboard...
echo Dashboard will open at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server.
echo.
streamlit run app.py --server.port 8501 --server.headless false
pause
