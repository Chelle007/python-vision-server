@echo off
REM Run on Windows from python-vision-server (venv already created once).
cd /d "%~dp0\.."

if not exist "models\escape_gestures.keras" (
  echo Missing models\escape_gestures.keras — copy the trained model here first.
  exit /b 1
)

if not exist ".venv\Scripts\activate.bat" (
  echo No .venv found. Create it first:
  echo   py -3.11 -m venv .venv
  echo   .venv\Scripts\activate
  echo   pip install -r requirements.txt
  echo   pip install -e .
  echo   pip install pyinstaller
  exit /b 1
)

call .venv\Scripts\activate.bat
pip install -q pyinstaller
pyinstaller --noconfirm pack\vision-server.spec

echo.
echo Done. Copy dist\VisionServer next to the Unity game .exe
echo so the folder looks like:
echo   YourGame.exe
echo   YourGame_Data\
echo   VisionServer\vision-server.exe
exit /b 0
