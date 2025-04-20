@echo off
echo Starting the pose detection system...

REM Create frontend_dis directory if it doesn't exist
if not exist "frontend_dis" mkdir frontend_dis

REM Activate the Python environment if needed
call backend_process/dependencies/scripts/activate

REM Install websockets library if needed
pip install websockets

REM Initialize variables
set video_option=
set delay_option=
set loop_option=
set framerate_option=
set input_choice=

REM Ask user if they want to use webcam or video file
echo.
echo Choose input source:
echo 1. Webcam (default)
echo 2. Video file
set /p input_choice="Enter your choice (1/2): "

REM Check explicitly for option 2, default to webcam for any other input
if /i "%input_choice%"=="2" goto :video_mode
goto :webcam_mode

:video_mode
echo.
echo Enter a video name or path:
echo.
echo Simple option: Just type the video name (example: video)
echo The system will search for it automatically
echo.
echo Full path option: Type the path if needed
echo.

echo Available videos:

REM Check for video files in common locations
if exist "*.mp4" (
    echo - Current directory:
    dir /b *.mp4
    echo.
)

if exist "backend_process\*.mp4" (
    echo - Backend_process directory:
    dir /b backend_process\*.mp4
    echo.
)

if exist "backend_process\scripts\*.mp4" (
    echo - Scripts directory:
    dir /b backend_process\scripts\*.mp4
    echo.
)

if exist "backend_process\videos\*.mp4" (
    echo - Videos directory:
    dir /b backend_process\videos\*.mp4
    echo.
)

if exist "videos\*.mp4" (
    echo - Root videos directory:
    dir /b videos\*.mp4
    echo.
)

echo.
set /p video_file="Enter video name or path: "

REM Ensure the path is properly quoted to handle spaces
set video_option=--video "%video_file%"

echo.
echo For video playback speed:
echo - Lower values = faster playback (1ms is fastest)
echo - Recommended: 1-5ms for best performance
set /p delay_val="Enter frame delay in ms (default: 1): "
if not "%delay_val%"=="" set delay_option=--delay %delay_val%
if "%delay_val%"=="" set delay_option=--delay 1

set /p loop_choice="Loop video when finished? (y/n): "
if /i "%loop_choice%"=="y" set loop_option=--loop

echo.
echo Frame rate control:
echo - 15: Fast processing
echo - 30: Default, balanced
echo - 60: More detailed
set /p framerate="Enter target frame rate (default: 30): "
if not "%framerate%"=="" set framerate_option=--frame-rate %framerate%
if "%framerate%"=="" set framerate_option=--frame-rate 30

echo.
echo Debug mode slows video but ensures model updates properly.
set /p debug_mode="Enable debug mode? (y/n): "
if /i "%debug_mode%"=="y" (
    set delay_option=--delay 20
    set framerate_option=--frame-rate 15
    echo Debug mode enabled: Slower playback for better updates
)
goto :start_program

:webcam_mode
echo Using webcam as input source...

:start_program
REM Start the Python capture script with selected options
echo.
echo Starting Python script with options: %video_option% %delay_option% %loop_option% %framerate_option%
start cmd /k "python backend_process/scripts/capture.py %video_option% %delay_option% %loop_option% %framerate_option%"

REM Try to open the frontend in VS Code Live Server if it's installed
echo Attempting to open frontend with Live Server...
code --goto frontend_dis/index.html:1 --reuse-window

echo.
echo System started!
echo Python script is running in the separate window
echo WebSocket server is started within the Python script
echo.
echo Instructions for opening with Live Server in VS Code:
echo 1. Right-click on frontend_dis/index.html in VS Code
echo 2. Select "Open with Live Server"
echo.
echo To stop the system, close the Python script window 