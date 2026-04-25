@echo off
echo.
echo  ============================================
echo   MLOps Pipeline - Starting Prediction API
echo  ============================================
echo.
echo  API will be live at: http://localhost:8001
echo  Swagger UI at:       http://localhost:8001/docs
echo.
cd /d "%~dp0"
set PYTHONPATH=%~dp0
uvicorn api.app:app --reload --port 8001
pause
