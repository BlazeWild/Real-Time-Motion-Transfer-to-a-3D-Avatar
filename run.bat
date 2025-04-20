@echo off
echo Starting the pose detection system...

REM Create frontend_dis directory if it doesn't exist
if not exist "frontend_dis" mkdir frontend_dis

REM Activate the Python environment if needed
call backend_process/dependencies/scripts/activate

REM Install websockets library if needed
pip install websockets

REM Start the Python capture script
start cmd /k "python backend_process/scripts/capture.py"

REM Try to open the frontend in VS Code Live Server if it's installed
echo Attempting to open frontend with Live Server...
code --goto frontend_dis/index.html:1 --reuse-window

echo System started!
echo Python script is running in the separate window
echo WebSocket server is started within the Python script
echo.
echo Instructions for opening with Live Server in VS Code:
echo 1. Right-click on frontend_dis/index.html in VS Code
echo 2. Select "Open with Live Server"
echo 3. The page will open in your browser
echo.
echo To stop the system, close the Python script window 