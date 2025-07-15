# Real-Time 3D Motion Transfer to Avatar

[![Medium Blog](https://img.shields.io/badge/Medium-Blog-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@blazewild215/real-time-motion-capture-animating-your-3d-avatar-with-live-tracking-f5690fe150e5)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

A complete system for capturing human motion from webcam or video and transferring it to a 3D avatar in real-time using MediaPipe, DNN and Three.js.

Created by [Ashok BK](https://github.com/blazewild) and [Ashim Nepal](https://github.com/nepalashim)

## Demo

<div align="center">
  <img src=".github/assets/demo/DEMO.png" alt="System Demo" width="100%">
</div>

The system detects body movements from webcam or video input and transfers them in real-time to a 3D avatar. You can use your own ReadyPlayerMe avatar and switch between webcam and video file inputs.

## Features

- Real-time pose detection from webcam or video file input
- Custom NN model for pose correction and refinement
- 17-keypoint skeleton mapping from detected landmarks
- Kalman filtering for smoother motion
- 3D visualization using Three.js
- Real-time motion transfer to 3D avatar models
- WebSocket communication between detection and visualization components
- Video or webcam input options with easy configuration

## System Requirements

- Python 3.8+ with pip
- Web browser with WebGL support
- VS Code with Live Server extension (recommended for frontend)
- Webcam (for live capture) or video files (for pre-recorded motion)
- Internet connection (for loading avatar models)

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/real-time-avatar-motion.git
   cd real-time-avatar-motion
   ```

2. Create a virtual environment inside the backend_process folder:

   On Windows:

   ```
   mkdir -p backend_process\dependencies
   python -m venv backend_process\dependencies
   backend_process\dependencies\Scripts\activate
   ```

   On macOS/Linux:

   ```
   mkdir -p backend_process/dependencies
   python -m venv backend_process/dependencies
   source backend_process/dependencies/bin/activate
   ```

3. Install required Python packages:

   ```
   pip install -r requirements.txt
   ```

4. If you're using Windows, you can run the `run.bat` file to start the application.
   For Mac/Linux users, follow the manual startup process below.

## Quick Start (Windows)

1. Run the `run.bat` file by double-clicking or from command prompt:

   ```
   run.bat
   ```

2. Choose your input source:

   - Option 1: Webcam (default)
   - Option 2: Video file (you can provide just the video name like "video" and the system will find it automatically)

3. For video files, you can configure:

   - Playback speed (delay between frames)
   - Looping options
   - Frame rate for processing
   - Debug mode for better model updates

4. The system will:

   - Activate the virtual environment
   - Start the Python backend for pose detection
   - Open the frontend file in VS Code
   - Provide instructions for opening with Live Server

5. In VS Code, right-click on `frontend_dis/index.html` and select "Open with Live Server"

## Manual Startup

1. Activate the virtual environment:

   On Windows:

   ```
   backend_process\dependencies\Scripts\activate
   ```

   On macOS/Linux:

   ```
   source backend_process/dependencies/bin/activate
   ```

2. Start the Python backend:

   For webcam:

   ```
   python backend_process/scripts/capture.py
   ```

   For video file:

   ```
   python backend_process/scripts/capture.py --video "path_to_video.mp4" --delay 1 --frame-rate 30
   ```

3. Serve the frontend:

   - Using VS Code Live Server: Right-click on `frontend_dis/index.html` and select "Open with Live Server"
   - Or using Python's built-in server: `python -m http.server 8000 --directory frontend_dis`

## Using Your Own 3D Avatar

You can easily use your own custom avatar from ReadyPlayerMe:

1. Visit [ReadyPlayerMe](https://readyplayer.me/) and create your custom avatar
2. After creating your avatar, click "Download" and choose "glTF/GLB"
3. You can also just copy the URL from the share link (ends with .glb)
4. Open `frontend_dis/glb-model.js` in a text editor
5. Find line 45 with: `const modelPath = "https://models.readyplayer.me/67be034c9fab1c21c486eb14.glb";`
6. Replace the URL with your avatar's URL
7. Save the file and refresh the browser window

Example:

```javascript
// Replace this
const modelPath = "https://models.readyplayer.me/67be034c9fab1c21c486eb14.glb";

// With your avatar URL
const modelPath = "https://models.readyplayer.me/YOUR_AVATAR_ID.glb";
```


## Usage

1. Stand in front of your webcam (or use a video file), ensuring your full body is visible.
2. The application will detect your pose and display:

   - Top: MediaPipe pose detection output
   - Bottom: Processed 17-keypoint skeleton
   - Browser: 3D avatar following your movements

3. Keyboard controls:
   - `k`: Toggle Kalman filter for smoother movement
   - `d`: Toggle DNN correction
   - `q`: Quit the application
   - `s`: Save a screenshot

## Processing Pipeline

The system processes motion in several stages:

1. **MediaPipe Pose Detection**: Captures 33 pose world landmarks using Google's MediaPipe/Blazepose library
2. **Landmark Selection**: Extracts 12 essential keypoints from the 33 MediaPipe landmarks:
   - Shoulders, elbows, wrists
   - Hips, knees, ankles
3. **DNN Correction**: Applies a neural network to correct and refine keypoint positions for accurate depth
4. **Orientation Enrichment**: Calculates local quaternion for 8 joints to apply the longitudinal rotation
5. **17-Keypoint Mapping**: Creates a full skeleton by:
   - Adding calculated joints (hips center, spine, neck)
   - Organizing joints in a standard hierarchy
6. **Kalman Filtering**: Applies statistical smoothing to reduce jitter and improve motion quality
7. **3D Model Animation**: Transfers processed joint rotations to the avatar's skeleton

## Project Structure

- `backend_process/scripts/`
  - `capture.py` - Main entry point, handles webcam/video capture and UI display
  - `processing.py` - Core processing logic for keypoint extraction and visualization
  - `quat_cal.py` - Handles quaternion calculations for rotational data
- `backend_process/dependencies/` - Python virtual environment (created during setup)
- `backend_process/models/` - Directory for model files (`dnn_model.pth`)
- `backend_process/videos/` - Place for storing video files (created automatically)
- `websocket_server.py` - Handles WebSocket communication with the frontend
- `video_websocket.py` - Handles streaming video frames to frontend
- `frontend_dis/` - Frontend files for 3D visualization:
  - `index.html` - Main frontend page
  - `canva.js` - Canvas and Three.js initialization
  - `glb-model.js` - 3D model handling and animation
  - `live-reload.js` - Auto-refresh functionality for development
- `run.bat` - Windows batch file for easy startup
- `videos/` - Alternative location for video files

## Technical Details

### Neural Network Architecture

The NN model consists of a multi-layer perceptron with the following architecture:

- Input: 36 values (12 keypoints × 3 coordinates)
- Hidden layers: 72 → 64 → 50 → 54 neurons
- Output: 36 values (12 corrected keypoints × 3 coordinates)

### 17-Keypoint Skeleton

The system maps MediaPipe's output to a 17-keypoint skeleton including:

- Hips (center)
- Spine (3 points)
- Head and neck
- Arms and hands (6 points)
- Legs and feet (6 points)

### Kalman Filtering

We implement a Kalman filter for each keypoint to reduce noise and jitter:

- State variables: Position (x,y,z) and velocity
- Observation: Raw keypoint positions
- Process noise and measurement covariance are tuned for smooth motion

### WebSocket Communication

- The backend sends 17-keypoint data and DNN status via WebSocket (port 8765)
- Video frames are streamed via a separate WebSocket (port 8766)
- The frontend receives this data and applies it to the 3D model
- 50Hz update rate for real-time performance

## Troubleshooting

Common issues:

- **No video feed**: Check if your webcam is connected and accessible
- **Poor detection**: Ensure good lighting and that your full body is visible
- **No model movement**: Check WebSocket connection status in browser console
- **NN correction fails**: Verify the model file exists in `backend_process/models/`
- **Missing dependencies**: Make sure the virtual environment is activated and all packages are installed
- **Live Server not refreshing**: Use the buttons in VS Code or toggle focus on the window

If Live Server isn't auto-refreshing:

1. Make sure the `live-reload.js` script is loaded in the HTML
2. Try clicking into another application window and back
3. Manually refresh once to trigger the auto-refresh mechanism

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgements

- [MediaPipe](https://github.com/google/mediapipe) for pose detection
- [PyTorch](https://pytorch.org/) for neural network implementation
- [Three.js](https://threejs.org/) for 3D visualization
- [ReadyPlayerMe](https://readyplayer.me/) for 3D avatar models
