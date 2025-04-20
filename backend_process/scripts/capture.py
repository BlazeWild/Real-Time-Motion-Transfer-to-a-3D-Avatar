import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import os
import json
import threading
import torch
import argparse

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
    
    # Also try to import video websocket server
    try:
        from backend_process import video_websocket
        USE_VIDEO_WEBSOCKET = True
        print("Video WebSocket server found, will stream video frames")
    except ImportError:
        USE_VIDEO_WEBSOCKET = False
        print("Video WebSocket server not found, no video frames will be streamed")
    
except ImportError:
    try:
        # Then try to import from root directory
        import websocket_server
        USE_WEBSOCKETS = True
        print("WebSocket server found in root directory, will use WebSockets to send keypoints")
        
        # Also try to import video websocket server from root
        try:
            import video_websocket
            USE_VIDEO_WEBSOCKET = True
            print("Video WebSocket server found in root, will stream video frames")
        except ImportError:
            USE_VIDEO_WEBSOCKET = False
            print("Video WebSocket server not found in root, no video frames will be streamed")
        
    except ImportError:
        USE_WEBSOCKETS = False
        USE_VIDEO_WEBSOCKET = False
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

# If video websocket is available, start the server in a separate thread
if USE_VIDEO_WEBSOCKET:
    def start_video_websocket_server():
        import asyncio
        asyncio.run(video_websocket.main())
    
    # Start the video websocket server in a separate thread
    video_websocket_thread = threading.Thread(target=start_video_websocket_server, daemon=True)
    video_websocket_thread.start()
    print("Video WebSocket server started in background thread")

# Print device info
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using CPU for all tensor operations")

# Set up argument parser
parser = argparse.ArgumentParser(description='Process video for pose detection and keypoint extraction')
parser.add_argument('--video', type=str, help='Path to video file. If not provided, webcam will be used.')
parser.add_argument('--delay', type=int, default=1, help='Delay between frames in milliseconds (for video files only). Higher values make video play slower.')
parser.add_argument('--loop', action='store_true', help='Loop the video file when it reaches the end.')
parser.add_argument('--frame-rate', type=int, default=30, help='Target frame rate for video processing. Default: 30 fps')
args = parser.parse_args()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize video source
if args.video:
    # Handle both relative and absolute paths
    video_path = args.video
    
    # Print the raw path provided for debugging
    print(f"Raw video name/path provided: {video_path}")
    
    # Remove any quotes or whitespace that might have been included in the path
    if video_path.startswith('"') and video_path.endswith('"'):
        video_path = video_path[1:-1]  # Remove surrounding quotes
    elif video_path.startswith("'") and video_path.endswith("'"):
        video_path = video_path[1:-1]  # Remove surrounding quotes
    
    video_path = video_path.strip()
    
    # If no path separators in the input, assume it's just a filename and search for it
    if not os.path.sep in video_path and not ('/' in video_path):
        print(f"Searching for video file named: {video_path}")
        
        # Common locations to search for videos
        search_dirs = [
            os.path.dirname(os.path.abspath(__file__)),  # scripts directory
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "videos"),  # backend_process/videos
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "videos"),  # project_root/videos
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # backend_process
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  # project root
        ]
        
        found = False
        for directory in search_dirs:
            if os.path.exists(directory):
                potential_path = os.path.join(directory, video_path)
                
                # Check if the file exists directly
                if os.path.exists(potential_path):
                    video_path = potential_path
                    found = True
                    print(f"Found video at: {video_path}")
                    break
                
                # Check if the file exists with common video extensions
                for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                    if not video_path.endswith(ext):
                        potential_path_with_ext = os.path.join(directory, video_path + ext)
                        if os.path.exists(potential_path_with_ext):
                            video_path = potential_path_with_ext
                            found = True
                            print(f"Found video at: {video_path}")
                            break
                
                if found:
                    break
        
        if not found:
            print(f"Could not find a video named '{video_path}' in any of the common directories")
    
    # Handle both slash types by normalizing path (replace / with \\ on Windows)
    video_path = os.path.normpath(video_path)
    
    if not os.path.isabs(video_path):
        # If relative path, resolve it relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        video_path = os.path.join(script_dir, video_path)
    
    # Normalize the path again after joining
    video_path = os.path.normpath(video_path)
    
    print(f"Resolved video path: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
        print("\nTrying to list available video files:")
        
        # Try to look for video files in common locations
        search_dirs = [
            os.getcwd(),  # Current working directory
            os.path.dirname(os.path.abspath(__file__)),  # Script directory
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "videos"),  # videos folder in parent directory
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "videos")  # videos folder in project root
        ]
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        found_videos = False
        
        for directory in search_dirs:
            if os.path.exists(directory):
                print(f"\nChecking directory: {directory}")
                files = os.listdir(directory)
                video_files = [f for f in files if os.path.splitext(f)[1].lower() in video_extensions]
                
                if video_files:
                    found_videos = True
                    print("Found video files:")
                    for video in video_files:
                        print(f"  - {video}")
                        print(f"    Use path: {os.path.relpath(os.path.join(directory, video), os.path.dirname(os.path.abspath(__file__)))}")
        
        if not found_videos:
            print("No video files found in common locations.")
        
        print("\nTIP: Make sure to use the correct relative path from the script directory:")
        print("  - For videos in the root folder: ..\\..\\video.mp4")
        print("  - For videos in backend_process folder: ..\\video.mp4")
        print("  - For videos in videos folder in root: ..\\..\\videos\\video.mp4")
        print("  - For videos in scripts folder: video.mp4")
        
        # Fall back to webcam
        print("\nFalling back to webcam input!")
        cap = cv2.VideoCapture(0)
        # Set moderate resolution for webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        print(f"Using video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        # Check video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Print video info
        print(f"Video properties:")
        print(f"  - Resolution: {width}x{height}")
        print(f"  - FPS: {fps}")
        print(f"  - Total frames: {frame_count}")
        print(f"  - Duration: {frame_count/fps:.2f} seconds")
        print(f"  - Playback delay: {args.delay}ms per frame")
        
        # Calculate target resolution (maintain aspect ratio)
        target_width = 640  # Fixed width for better performance
        target_height = int(height * (target_width / width))
        print(f"  - Target resolution for processing: {target_width}x{target_height}")
else:
    print("Using webcam as video source")
    cap = cv2.VideoCapture(0)
    # Set moderate resolution to ensure good performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Smaller width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Smaller height

# Check if video source is opened and print resolution
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()
else:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video source opened with resolution: {width}x{height}")

# Print instructions
print("\nControls:")
print("  Press 'q' to quit")
print("  Press 'k' to toggle Kalman filter on/off")
print("  Press 'd' to toggle DNN correction on/off")
print("  Press 's' to save a screenshot")

# Ensure the frontend_dis directory exists
if not os.path.exists("frontend_dis"):
    os.makedirs("frontend_dis")

# Create videos directory in root if it doesn't exist
videos_root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "videos")
if not os.path.exists(videos_root_dir):
    os.makedirs(videos_root_dir)
    print(f"Created videos directory at: {videos_root_dir}")

# Create videos directory in backend_process if it doesn't exist
videos_backend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "videos")
if not os.path.exists(videos_backend_dir):
    os.makedirs(videos_backend_dir)
    print(f"Created videos directory at: {videos_backend_dir}")

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
    
    # For video files, calculate frame skipping based on target frame rate
    frame_skip = 1
    if args.video:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps > args.frame_rate:
            frame_skip = max(1, int(video_fps / args.frame_rate))
            print(f"Video FPS ({video_fps}) > target FPS ({args.frame_rate})")
            print(f"Processing every {frame_skip} frame(s) for better performance")
    
    frame_count = 0
    
    while cap.isOpened():
        # Capture frame-by-frame
        success, frame = cap.read()
        if not success:
            if args.video and args.loop:
                # Reset video to beginning for looping
                print("End of video reached. Looping back to start.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                continue
            else:
                print("End of video reached or error reading frame.")
                break
        
        frame_count += 1
        
        # Skip frames for video files based on calculated frame_skip
        if args.video and frame_count % frame_skip != 0:
            continue
        
        # Resize video frames for better performance
        if args.video:
            # Resize to target resolution
            frame = cv2.resize(frame, (target_width, target_height))
        
        # Flip the image horizontally for a selfie-view display (only for webcam)
        if not args.video:
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
                    
                    # Add DNN status to the keypoints dictionary
                    keypoints_dict["dnn_enabled"] = processing.use_dnn_correction
                    
                    # For video input, add a timestamp to force updates
                    if args.video:
                        keypoints_dict["timestamp"] = time.time()
                    
                    # Use WebSockets if available
                    if USE_WEBSOCKETS:
                        websocket_server.update_keypoints(keypoints_dict)
                        
                        # Print to console occasionally
                        if int(time.time()) % 5 == 0:  # Every 5 seconds
                            print(f"Keypoints sent via WebSocket: {len(keypoints_dict)} points")
                            if args.video:
                                print(f"Video mode active, playing frame {frame_count}")
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
                
                # Send stacked view to video websocket if available
                if USE_VIDEO_WEBSOCKET:
                    # This will encode and send the frame to connected clients
                    video_websocket.update_frame(stacked_view)
                
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
        
        # Adjust wait time based on delay parameter for videos
        # Lower delay for better performance with video files
        wait_time = min(args.delay, 1) if args.video else 5
        
        # For video input, force more frequent WebSocket updates
        if args.video and USE_WEBSOCKETS and 'keypoints_dict' in locals() and keypoints_dict:
            # Send keypoint updates every single processed frame for video
            websocket_server.update_keypoints(keypoints_dict)
            
            # Print debug info occasionally
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: Forced keypoint update sent")
        
        # Press 'q' to exit, 'k' to toggle Kalman filter, 'd' to toggle DNN correction, 's' to save screenshot
        key = cv2.waitKey(wait_time) & 0xFF
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