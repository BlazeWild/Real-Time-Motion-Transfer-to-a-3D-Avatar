import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import os
import json
import threading
import torch

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the processing module with the correct path
from backend_process.scripts import processing

# Try to import the websocket server module
try:
    # First, try to import from backend_process directory
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend_process import websocket_server
    USE_WEBSOCKETS = True
    print("WebSocket server found in backend_process directory, will use WebSockets to send keypoints")
except ImportError:
    try:
        # Then try to import from root directory
        import websocket_server
        USE_WEBSOCKETS = True
        print("WebSocket server found in root directory, will use WebSockets to send keypoints")
    except ImportError:
        USE_WEBSOCKETS = False
        print("WebSocket server module not found, no keypoints will be shared")

# If websocket is available, start the server in a separate thread
if USE_WEBSOCKETS:
    def start_websocket_server():
        import asyncio
        asyncio.run(websocket_server.main())
    
    # Start the websocket server in a separate thread
    websocket_thread = threading.Thread(target=start_websocket_server, daemon=True)
    websocket_thread.start()
    print("WebSocket server started in background thread")

# Print device info
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using CPU for all tensor operations")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set moderate resolution to ensure good performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Smaller width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Smaller height

# Check if camera is opened and print resolution
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
else:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened with resolution: {width}x{height}")

# Print instructions
print("\nControls:")
print("  Press 'q' to quit")
print("  Press 'k' to toggle Kalman filter on/off")
print("  Press 'd' to toggle DNN correction on/off")
print("  Press 's' to save a screenshot")

# Ensure the frontend_dis directory exists
if not os.path.exists("frontend_dis"):
    os.makedirs("frontend_dis")

# Configure BlazePose with world landmarks
with mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    enable_segmentation=True,
) as pose:
    
    prev_frame_time = 0
    curr_frame_time = 0
    fps_values = []  # Store recent FPS values for smoothing
    fps_smooth = 0
    
    while cap.isOpened():
        # Capture frame-by-frame
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
            
        # Flip the image horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        
        # To improve performance, optionally mark the image as not writeable
        frame.flags.writeable = False
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect pose
        results = pose.process(image_rgb)
        
        # Convert RGB back to BGR for OpenCV
        frame.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            # Draw pose landmarks on the image
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Calculate and display FPS
            curr_frame_time = time.time()
            fps = 1 / (curr_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = curr_frame_time
            
            # Smooth FPS display using rolling average (last 10 frames)
            fps_values.append(fps)
            if len(fps_values) > 10:
                fps_values.pop(0)
            fps_smooth = sum(fps_values) / len(fps_values)
            
            # Add FPS display to the image
            fps_text = f"FPS: {fps_smooth:.1f}"
            
            # Get world landmarks (3D coordinates)
            if results.pose_world_landmarks:
                world_landmarks = results.pose_world_landmarks.landmark
                
                # Get mapped keypoints only (not the visualization from processing.py)
                # We'll create our own vertical display here
                _, mapped_keypoints = processing.process_frame(image, world_landmarks, return_keypoints=True)
                
                # Process the keypoints data for WebSockets
                if mapped_keypoints:
                    keypoints_dict = {}
                    for i, name in enumerate(processing.OUTPUT_KEYPOINT_ORDER):
                        if i < len(mapped_keypoints):
                            keypoints_dict[name] = mapped_keypoints[i]
                    
                    # Use WebSockets if available
                    if USE_WEBSOCKETS:
                        websocket_server.update_keypoints(keypoints_dict)
                        
                        # Print to console occasionally
                        if int(time.time()) % 5 == 0:  # Every 5 seconds
                            print(f"Keypoints sent via WebSocket")
                    else:
                        # Just log that we detected keypoints but not sharing them
                        if int(time.time()) % 5 == 0:  # Every 5 seconds
                            print(f"Keypoints detected but not shared (WebSocket not available)")
                
                # Create our own vertical stacked display (MediaPipe on top, skeleton on bottom)
                h, w = image.shape[:2]
                
                # Create blank canvas for bottom side with mapped skeleton
                bottom_view = np.zeros((h, w, 3), dtype=np.uint8)
                bottom_view[:] = (50, 50, 50)  # dark gray background
                
                # Draw skeleton on bottom view if we have valid keypoints
                if mapped_keypoints:
                    # Draw mapped keypoints on the bottom view
                    bottom_view = processing.draw_skeleton(bottom_view, mapped_keypoints, (0, 0, 255), thickness=3)
                else:
                    # Message if mapping failed
                    cv2.putText(bottom_view, "Mapping failed - Check body position", (50, h//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Create the vertical stacked view
                stacked_view = np.zeros((h*2, w, 3), dtype=np.uint8)
                
                # Add FPS to the MediaPipe view
                cv2.putText(image, fps_text, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Add title to each view
                cv2.putText(image, "MediaPipe Pose", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(bottom_view, "Mapped 17-Keypoint Skeleton", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                
                # Add controls info at bottom of skeleton view
                cv2.putText(bottom_view, "Controls: 'k'=toggle Kalman | 'd'=toggle DNN", (10, h-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Copy views to the stacked image
                stacked_view[:h, :] = image  # Top half is MediaPipe
                stacked_view[h:, :] = bottom_view  # Bottom half is skeleton
                
                # Create a smaller window for display (scale down if needed)
                display_scale = 0.9
                display_width = int(stacked_view.shape[1] * display_scale)
                display_height = int(stacked_view.shape[0] * display_scale)
                display_image = cv2.resize(stacked_view, (display_width, display_height))
                
                # Show the vertical stacked view
                cv2.imshow('MediaPipe and Mapped Keypoints (Vertical View)', display_image)
            
            else:
                # If no world landmarks, just show the regular MediaPipe output
                cv2.putText(image, fps_text, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                display_image = cv2.resize(image, (int(image.shape[1]*display_scale), int(image.shape[0]*display_scale)))
                cv2.imshow('MediaPipe Pose', display_image)
        else:
            # If no pose detected, just show the regular frame
            cv2.putText(frame, "No pose detected", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            display_image = cv2.resize(frame, (int(frame.shape[1]*0.9), int(frame.shape[0]*0.9)))
            cv2.imshow('MediaPipe Pose', display_image)
        
        # Press 'q' to exit, 'k' to toggle Kalman filter, 'd' to toggle DNN correction, 's' to save screenshot
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('k'):
            # Toggle Kalman filter
            kalman_state = processing.kalman_filter.toggle()
            print(f"Kalman filter {'enabled' if kalman_state else 'disabled'}")
        elif key == ord('d'):
            # Toggle DNN correction
            dnn_state = processing.toggle_dnn_correction()
            print(f"DNN correction {'enabled' if dnn_state else 'disabled'}")
        elif key == ord('s'):
            # Save screenshot
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"pose_screenshot_{timestamp}.jpg"
            if 'display_image' in locals():
                cv2.imwrite(filename, display_image)
                print(f"Screenshot saved as {filename}")

# Release resources
cap.release()
cv2.destroyAllWindows() 