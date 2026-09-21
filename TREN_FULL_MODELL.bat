@echo off
setlocal
cd /d "%~dp0"
"C:\Users\hat\AppData\Local\Programs\Python\Python313\python.exe" -u train_full_model.py --epochs 12 --batch-size 48 --workers 4
echo.
echo Trening ferdig. Se full_training_run\summary.json.
pause
