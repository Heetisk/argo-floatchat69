@echo off
REM Activate virtual environment and run Streamlit app
echo Starting Argo Data Explorer...
call .venv\Scripts\activate.bat
streamlit run app.py
pause

