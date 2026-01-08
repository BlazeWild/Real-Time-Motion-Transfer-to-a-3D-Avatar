import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import sys

# Try importing mediapipe with different methods
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    # Fallback for newer mediapipe versions
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing

# Add the project root to the Python path to allow imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Explicitly set device to CPU
device = torch.device('cpu')
print(f"Using device: {device}")

# Kalman filter for smoothing keypoints
class KeypointKalmanFilter:
    def __init__(self, num_keypoints, dim_per_point=3, use_kalman=True):
        self.num_keypoints = num_keypoints
        self.dim_per_point = dim_per_point
        self.dim_state = num_keypoints * dim_per_point
        self.use_kalman = use_kalman
        
        # Initialize Kalman filters (one per keypoint)
        self.filters = []
        for _ in range(num_keypoints):
            kf = cv2.KalmanFilter(dim_per_point * 2, dim_per_point)  # State: [x,y,z,dx,dy,dz], Measurement: [x,y,z]
            
            # Transition matrix (physical model)
            kf.transitionMatrix = np.eye(dim_per_point * 2, dtype=np.float32)
            for i in range(dim_per_point):
                kf.transitionMatrix[i, i + dim_per_point] = 0.2  # Add velocity component (reduced weight)
            
            # Measurement matrix
            kf.measurementMatrix = np.zeros((dim_per_point, dim_per_point * 2), dtype=np.float32)
            for i in range(dim_per_point):
                kf.measurementMatrix[i, i] = 1.0
            
            # Process noise covariance - lower for smoother movement
            kf.processNoiseCov = np.eye(dim_per_point * 2, dtype=np.float32) * 1e-4
            
            # Measurement noise covariance - higher for smoother movement
            kf.measurementNoiseCov = np.eye(dim_per_point, dtype=np.float32) * 1e-2
            
            # Initial state
            kf.statePost = np.zeros((dim_per_point * 2, 1), dtype=np.float32)
            
            self.filters.append(kf)
        
        self.initialized = False
    
    def update(self, keypoints):
        """Update the filter with new keypoint positions"""
        if not self.use_kalman:
            return keypoints
            
        if not self.initialized:
            # Initialize the filters with first measurements
            for i, kp in enumerate(keypoints):
                self.filters[i].statePost[:self.dim_per_point] = np.array(kp, dtype=np.float32).reshape(-1, 1)
            self.initialized = True
            return keypoints
        
        # Apply Kalman filter updates and predictions
        filtered_keypoints = []
        for i, kp in enumerate(keypoints):
            # Convert keypoint to measurement
            measurement = np.array(kp, dtype=np.float32).reshape(-1, 1)
            
            # Prediction step
            predicted = self.filters[i].predict()
            
            # Correction step
            corrected = self.filters[i].correct(measurement)
            
            # Get smoothed coordinates (first dim_per_point elements of the state)
            smoothed_kp = corrected[:self.dim_per_point].flatten().tolist()
            filtered_keypoints.append(smoothed_kp)
        
        return filtered_keypoints

    def toggle(self):
        """Toggle Kalman filter on/off"""
        self.use_kalman = not self.use_kalman
        return self.use_kalman

# Initialize Kalman filter for the 17 output keypoints with improved parameters
kalman_filter = KeypointKalmanFilter(num_keypoints=17, dim_per_point=3, use_kalman=True)

# Mapping of MediaPipe landmarks to our simplified format
# We only need a subset of the 33 MediaPipe keypoints, i.e 12 keypoints needed step1:
REQUIRED_KEYPOINTS = {
    11: "LeftArm",       # LEFT_SHOULDER
    12: "RightArm",      # RIGHT_SHOULDER
    13: "LeftForeArm",   # LEFT_ELBOW  
    14: "RightForeArm",  # RIGHT_ELBOW
    15: "LeftHand",      # LEFT_WRIST
    16: "RightHand",     # RIGHT_WRIST
    23: "LeftUpLeg",     # LEFT_HIP
    24: "RightUpLeg",    # RIGHT_HIP
    25: "LeftLeg",       # LEFT_KNEE
    26: "RightLeg",      # RIGHT_KNEE
    27: "LeftFoot",      # LEFT_ANKLE
    28: "RightFoot"      # RIGHT_ANKLE
}

# Order of the 17 keypoints expected by the model
OUTPUT_KEYPOINT_ORDER = [
    "Hips", "Spine1", "Spine2", "Neck", "Head",
    "LeftArm", "LeftForeArm", "LeftHand",
    "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot",
    "RightUpLeg", "RightLeg", "RightFoot"
]

# DNN model for correcting keypoints - updated to match saved model architecture
class DNN(nn.Module):
    def __init__(self, input_size=36, hidden1_size=72, hidden2_size=64, hidden3_size=50, hidden4_size=54, output_size=36):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, hidden3_size)
        self.fc4 = nn.Linear(hidden3_size, hidden4_size)
        self.fc5 = nn.Linear(hidden4_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Path to the DNN model - fixed path to find the model in the correct directory
dnn_modelpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "dnn_model.pth")
print(f"Looking for DNN model at: {dnn_modelpath}")

# Colors for visualization
MEDIAPIPE_COLOR = (0, 255, 0)  # Green
MAPPED_COLOR = (0, 0, 255)     # Red
BACKGROUND_COLOR = (50, 50, 50)  # Dark gray background for better visibility

# Initialize DNN model - will be loaded if model file exists
model = None

# Enable/disable flags for different processing steps
use_dnn_correction = True  # Flag to toggle DNN correction

def log_error(message):
    """Simple logging function for errors"""
    print(f"ERROR: {message}")

def initialize_model():
    """Initialize the DNN model if the model file exists"""
    global model
    
    # Create model with the correct architecture to match saved weights
    model = DNN(input_size=36, hidden1_size=72, hidden2_size=64, 
                hidden3_size=50, hidden4_size=54, output_size=36)
    
    # Ensure model uses CPU
    model = model.to(device)
    
    # Load weights if model file exists
    if os.path.exists(dnn_modelpath):
        try:
            # Explicitly load to CPU
            model.load_state_dict(torch.load(dnn_modelpath, map_location=device))
            model.eval()
            print(f"Loaded DNN model from {dnn_modelpath} on {device}")
        except Exception as e:
            log_error(f"Failed to load DNN model: {e}")
            model = None
    else:
        log_error(f"Model file {dnn_modelpath} not found. Proceeding without DNN correction.")
        model = None

def extract_required_keypoints(world_landmarks):
    """Extract the required keypoints from MediaPipe world landmarks"""
    filtered_keypoints = {}
    
    for idx, name in REQUIRED_KEYPOINTS.items():
        if idx < len(world_landmarks):
            landmark = world_landmarks[idx]
            # Keep original coordinates - no need to flip here
            filtered_keypoints[name] = [landmark.x, landmark.y, landmark.z]
    
    return filtered_keypoints

def correct_keypoints_with_dnn(filtered_keypoints):
    """Apply DNN correction to the filtered keypoints if model is available"""
    if model is None:
        return filtered_keypoints
    
    # Prepare input tensor - ensure we have exactly 12 keypoints (36 values)
    keypoint_list = []
    for name in REQUIRED_KEYPOINTS.values():
        if name in filtered_keypoints:
            keypoint_list.extend(filtered_keypoints[name])
        else:
            keypoint_list.extend([0, 0, 0])  # Fill missing keypoints with zeros
    
    # Ensure we have exactly 36 values (12 keypoints * 3 coordinates)
    if len(keypoint_list) != 36:
        log_error(f"Expected 36 input values, got {len(keypoint_list)}. Skipping DNN correction.")
        return filtered_keypoints
    
    # Create tensor and explicitly move to CPU
    input_tensor = torch.tensor(keypoint_list, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor).squeeze(0).tolist()
    
    # Ensure we got 36 output values (12 keypoints * 3 coordinates)
    if len(output) != 36:
        log_error(f"Expected 36 output values from DNN, got {len(output)}. Using original keypoints.")
        return filtered_keypoints
    
    # Update the filtered keypoints with corrected values
    corrected_keypoints = {}
    i = 0
    for name in REQUIRED_KEYPOINTS.values():
        corrected_keypoints[name] = [output[i], output[i+1], output[i+2]]
        i += 3
    
    return corrected_keypoints

def map_to_output_keypoints(filtered_keypoints):
    """Map the filtered keypoints to the 17 required output keypoints"""
    # Extract keypoints
    left_shoulder = filtered_keypoints.get('LeftArm')
    right_shoulder = filtered_keypoints.get('RightArm')
    left_elbow = filtered_keypoints.get('LeftForeArm')
    right_elbow = filtered_keypoints.get('RightForeArm')
    left_wrist = filtered_keypoints.get('LeftHand')
    right_wrist = filtered_keypoints.get('RightHand')
    left_hip = filtered_keypoints.get('LeftUpLeg')
    right_hip = filtered_keypoints.get('RightUpLeg')
    left_knee = filtered_keypoints.get('LeftLeg')
    right_knee = filtered_keypoints.get('RightLeg')
    left_ankle = filtered_keypoints.get('LeftFoot')
    right_ankle = filtered_keypoints.get('RightFoot')
    
    # Check if we have all required keypoints
    required_points = [left_shoulder, right_shoulder, left_hip, right_hip]
    if any(point is None for point in required_points):
        log_error("Missing essential keypoints")
        return None
    
    # Calculate derived keypoints
    # Hip center (midpoint between left and right hip)
    hips = [(left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2,
            (left_hip[2] + right_hip[2]) / 2]
    
    # Neck (midpoint between shoulders)
    neck = [(left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2,
            (left_shoulder[2] + right_shoulder[2]) / 2]
    
    # Spine1 (1/3 up from hips to neck)
    spine1 = [hips[0] + (neck[0] - hips[0]) * 0.33,
              hips[1] + (neck[1] - hips[1]) * 0.33,
              hips[2] + (neck[2] - hips[2]) * 0.33]
    
    # Spine2 (2/3 up from hips to neck)
    spine2 = [hips[0] + (neck[0] - hips[0]) * 0.66,
              hips[1] + (neck[1] - hips[1]) * 0.66,
              hips[2] + (neck[2] - hips[2]) * 0.66]
    
    # Head (extrapolate above neck by same distance as neck to spine2)
    head_offset = [neck[0] - spine2[0], neck[1] - spine2[1], neck[2] - spine2[2]]
    head = [neck[0] + head_offset[0], neck[1] + head_offset[1], neck[2] + head_offset[2]]
    
    # Construct the 17 keypoints in the required order
    result = [
        hips,                # 0: Hips
        spine1,              # 1: Spine1
        spine2,              # 2: Spine2
        neck,                # 3: Neck
        head,                # 4: Head
        left_shoulder,       # 5: LeftArm
        left_elbow,          # 6: LeftForeArm
        left_wrist,          # 7: LeftHand
        right_shoulder,      # 8: RightArm
        right_elbow,         # 9: RightForeArm
        right_wrist,         # 10: RightHand
        left_hip,            # 11: LeftUpLeg
        left_knee,           # 12: LeftLeg
        left_ankle,          # 13: LeftFoot
        right_hip,           # 14: RightUpLeg
        right_knee,          # 15: RightLeg
        right_ankle          # 16: RightFoot
    ]
    
    return result

def draw_skeleton(image, keypoints, color, thickness=2):
    """Draw skeleton on the image based on keypoints"""
    h, w = image.shape[:2]
    
    # Start with a dark background for better visibility
    image[:] = BACKGROUND_COLOR
    
    # Scale factor from 3D world coordinates to 2D image coordinates
    scale_x = w * 0.4  # Use more of the image width
    scale_y = h * 0.4  # Use more of the image height
    
    # Function to convert 3D world coordinates to 2D image coordinates
    def world_to_image(point):
        # No need to flip x anymore since we fixed orientation in mapped keypoints
        img_x = int(w/2 + point[0] * scale_x)  # Use original x
        img_y = int(h/2 - point[1] * scale_y)  # Keep y inverted for screen coordinates
        return (img_x, img_y)
    
    # Convert 3D points to 2D image points
    points_2d = [world_to_image(kp) for kp in keypoints]
    
    # Draw keypoints
    radius = 6
    for i, pt in enumerate(points_2d):
        # Use different colors for key points
        if i < 5:  # Torso points (hips, spine, neck, head)
            circle_color = (255, 200, 0)  # Yellow-orange
        elif i < 11:  # Arm points
            circle_color = (0, 200, 255)  # Light blue
        else:  # Leg points
            circle_color = (200, 0, 200)  # Purple
            
        cv2.circle(image, pt, radius, circle_color, -1)
        
        # Add keypoint labels
        if i in [0, 3, 5, 8, 11, 14]:  # Only label key points to avoid clutter
            label = OUTPUT_KEYPOINT_ORDER[i]
            cv2.putText(image, label, (pt[0] + 5, pt[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Define connections between keypoints for skeleton
    connections = [
        (0, 1),    # Hips to Spine1
        (1, 2),    # Spine1 to Spine2
        (2, 3),    # Spine2 to Neck
        (3, 4),    # Neck to Head
        (3, 5),    # Neck to LeftArm
        (5, 6),    # LeftArm to LeftForeArm
        (6, 7),    # LeftForeArm to LeftHand
        (3, 8),    # Neck to RightArm
        (8, 9),    # RightArm to RightForeArm
        (9, 10),   # RightForeArm to RightHand
        (0, 11),   # Hips to LeftUpLeg
        (11, 12),  # LeftUpLeg to LeftLeg
        (12, 13),  # LeftLeg to LeftFoot
        (0, 14),   # Hips to RightUpLeg
        (14, 15),  # RightUpLeg to RightLeg
        (15, 16)   # RightLeg to RightFoot
    ]
    
    # Draw connecting lines
    for connection in connections:
        cv2.line(image, points_2d[connection[0]], points_2d[connection[1]], color, thickness)
    
    return image

def toggle_dnn_correction():
    """Toggle DNN correction on/off"""
    global use_dnn_correction
    use_dnn_correction = not use_dnn_correction
    return use_dnn_correction

def process_frame(frame, world_landmarks, return_keypoints=False):
    """
    Process a frame of MediaPipe landmarks to produce a side-by-side comparison
    If return_keypoints is True, also return the 17 processed keypoints
    """
    # Initialize model on first call
    global model
    if model is None:
        initialize_model()
    
    # Extract required keypoints from MediaPipe landmarks
    filtered_keypoints = extract_required_keypoints(world_landmarks)
    
    # Apply DNN correction if enabled and available
    if use_dnn_correction and model is not None:
        filtered_keypoints = correct_keypoints_with_dnn(filtered_keypoints)
    
    # Map to output keypoints format (17 keypoints)
    mapped_keypoints = map_to_output_keypoints(filtered_keypoints)
    
    # Apply Kalman filtering if enabled
    mapped_keypoints = kalman_filter.update(mapped_keypoints)
    
    # Create side-by-side view
    side_by_side = create_side_by_side_view(frame, mapped_keypoints)
    
    if return_keypoints:
        return side_by_side, mapped_keypoints
    return side_by_side

def create_side_by_side_view(frame, mapped_keypoints):
    """Create a vertical view with original frame on top and mapped keypoints on bottom"""
    # Create a blank stacked image (changed from side-by-side to vertical)
    h, w = frame.shape[:2]
    # Now we stack vertically, so height is doubled
    stacked_view = np.zeros((h*2, w, 3), dtype=np.uint8)
    
    # Copy original frame to the top
    stacked_view[:h, :] = frame
    
    # Create a blank canvas for the bottom side with dark background
    bottom_side = np.zeros((h, w, 3), dtype=np.uint8)
    bottom_side[:] = BACKGROUND_COLOR  # Set background color
    
    # Status panel background (dark panel with border)
    status_panel = np.zeros((180, 300, 3), dtype=np.uint8)
    status_panel[:, :] = (40, 40, 40)  # Dark gray background
    cv2.rectangle(status_panel, (0, 0), (299, 179), (100, 100, 100), 1)  # Border
    
    # Title for status panel
    cv2.putText(status_panel, "Processing Pipeline Status", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    cv2.line(status_panel, (10, 35), (290, 35), (100, 100, 100), 1)
    
    # Track status of each step
    steps_status = []
    
    # Step 1: Keypoint extraction
    steps_status.append({
        "name": "1. MediaPipe Landmark Selection",
        "status": True,
        "color": (0, 255, 0)  # Green
    })
    
    # Step 2: DNN Correction status
    if model is not None and use_dnn_correction:
        steps_status.append({
            "name": "2. Neural Network Correction",
            "status": True,
            "color": (0, 255, 0)  # Green
        })
    else:
        status_color = (0, 165, 255)  # Orange for disabled
        status_message = "2. Neural Network Correction"
        if model is None:
            status_message += " (model not found)"
        else:
            status_message += " (disabled)"
            
        steps_status.append({
            "name": status_message,
            "status": False,
            "color": status_color
        })
    
    # Step 3: Mapping status
    if mapped_keypoints:
        steps_status.append({
            "name": "3. 17-Keypoint Skeleton Mapping",
            "status": True,
            "color": (0, 255, 0)  # Green
        })
        
        # Step 4: Apply Kalman filter status
        if kalman_filter.use_kalman:
            steps_status.append({
                "name": "4. Kalman Filter Smoothing",
                "status": True,
                "color": (0, 255, 0)  # Green
            })
        else:
            steps_status.append({
                "name": "4. Kalman Filter Smoothing",
                "status": False,
                "color": (0, 165, 255)  # Orange
            })
        
        # Draw mapped keypoints skeleton on the bottom side
        bottom_side = draw_skeleton(bottom_side, mapped_keypoints, MAPPED_COLOR, thickness=3)
        
        # Add labels and information with better formatting
        # Draw a semi-transparent overlay for text background on top side
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, stacked_view[:h, :])
    else:
        # If mapping failed, add error message
        steps_status.append({
            "name": "3. 17-Keypoint Skeleton Mapping",
            "status": False,
            "color": (0, 0, 255)  # Red
        })
        
        # Add error message on the bottom side
        cv2.putText(bottom_side, "Mapping failed - Check body position", (50, 350), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Update status panel with all steps
    y_offset = 50
    for step in steps_status:
        check_mark = "✓" if step["status"] else "✗"
        cv2.putText(status_panel, f"{check_mark} {step['name']}", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, step["color"], 1)
        y_offset += 25
    
    # Place status panel on the bottom side
    bottom_side[20:20+status_panel.shape[0], 20:20+status_panel.shape[1]] = status_panel
    
    # Add clear titles
    cv2.putText(stacked_view[:h, :], "MediaPipe Pose", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(stacked_view[h:, :], "Mapped 17-Keypoint Skeleton", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Add key controls info at bottom
    cv2.putText(bottom_side, "Controls: 'k'=toggle Kalman | 'd'=toggle DNN", (10, h-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Copy bottom side to the stacked image
    stacked_view[h:, :] = bottom_side
    
    return stacked_view

# Initialize the model when module is imported
initialize_model()