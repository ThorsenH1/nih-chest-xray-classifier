@echo off
setlocal
cd /d "%~dp0"
"C:\Users\hat\AppData\Local\Programs\Python\Python313\python.exe" -u verify_model.py --workers 2 --batch-size 64 --output-dir audit_results
echo.
echo Kontroll ferdig. Se audit_results\summary.json og audit_results\per_disease_metrics.csv.
pause
