import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
from datetime import timedelta
import argparse
import os
import mediapipe as mp
import math

# Patch for PyTorch compatibility with Streamlit
import sys
from types import ModuleType

class PatchedModule(ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__dict__.update(sys.modules[name].__dict__)
    
    def __getattr__(self, attr):
        if attr == '__path__':
            class PatchedPath:
                _path = []
            return PatchedPath()
        return ModuleType.__getattr__(self, attr)

# Apply the patch to torch._classes
if 'torch._classes' in sys.modules:
    sys.modules['torch._classes'] = PatchedModule('torch._classes')

class ExamCheatingDetector:
    def __init__(self, video_path, output_dir="results", use_cuda=True, half_precision=False, batch_size=1):
        """
        Initialize the exam cheating detector.
        
        Args:
            video_path (str): Path to the input video file
            output_dir (str): Directory to save results
            use_cuda (bool): Whether to use CUDA GPU acceleration
            half_precision (bool): Whether to use FP16 half-precision
            batch_size (int): Batch size for inference
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        self.half_precision = half_precision
        self.batch_size = batch_size
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize detection models
        print(f"Loading detection models on {self.device}...")
        
        # Load object detector
        self.object_detector = YOLO("yolov8x.pt")
        self.object_detector.to(self.device)
        if self.half_precision and self.device == 'cuda':
            self.object_detector = self.object_detector.half()
        
        # Initialize MediaPipe Face Mesh for facial landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe Drawing utilities for visualization
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Classes of interest for cheating detection
        self.target_classes = {
            0: "person",    # Person detection
            67: "cell phone",
            73: "book",
            84: "book"      # Sometimes books are classified as different IDs
        }
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video file {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = self.total_frames / self.fps
        
        # Detection parameters
        self.frame_skip = int(self.fps / 2)  # Process every half second
        self.face_detection_interval = int(self.fps * 2)  # Face direction check every 2 seconds
        
        # We assume exactly 1 person should be in the frame by default
        self.expected_person_count = 1
        self.gaze_direction_baseline = None  # Will track initial gaze direction
        self.head_pose_baseline = None  # Will track initial head pose
        
        # Violation tracking
        self.violations = {
            "cell_phone": [],
            "additional_person": [],
            "book": [],
            "looking_away": [],
            "person_missing": []
        }

        # Time tracking
        self.processing_time = 0
        
        # Eye landmarks indices
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Face landmarks for head pose
        self.FACE_POSE_LANDMARKS = [33, 263, 1, 61, 291, 199]
        
        print(f"Video loaded: {self.total_frames} frames, {self.video_duration:.2f} seconds")
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def get_eye_position(self, landmarks, eye_indices):
        """Extract eye position from facial landmarks"""
        eye_points = [landmarks[i] for i in eye_indices]
        
        # Calculate eye center
        eye_center_x = sum(point.x for point in eye_points) / len(eye_points)
        eye_center_y = sum(point.y for point in eye_points) / len(eye_points)
        
        # Extract pupil (using center point - simplified approach)
        pupil_x, pupil_y = eye_center_x, eye_center_y
        
        return (pupil_x, pupil_y)
    
    def get_gaze_direction(self, landmarks):
        """Determine gaze direction based on eye landmarks"""
        if landmarks is None:
            return None
        
        # Get positions for both eyes
        left_eye_pos = self.get_eye_position(landmarks, self.LEFT_EYE)
        right_eye_pos = self.get_eye_position(landmarks, self.RIGHT_EYE)
        
        # Calculate eye centers
        left_eye_center = (sum(landmarks[i].x for i in self.LEFT_EYE) / len(self.LEFT_EYE),
                         sum(landmarks[i].y for i in self.LEFT_EYE) / len(self.LEFT_EYE))
        
        right_eye_center = (sum(landmarks[i].x for i in self.RIGHT_EYE) / len(self.RIGHT_EYE),
                          sum(landmarks[i].y for i in self.RIGHT_EYE) / len(self.RIGHT_EYE))
        
        # Calculate pupil position relative to eye center
        left_pupil_rel_x = left_eye_pos[0] - left_eye_center[0]
        left_pupil_rel_y = left_eye_pos[1] - left_eye_center[1]
        
        right_pupil_rel_x = right_eye_pos[0] - right_eye_center[0]
        right_pupil_rel_y = right_eye_pos[1] - right_eye_center[1]
        
        # Average the relative positions
        avg_pupil_rel_x = (left_pupil_rel_x + right_pupil_rel_x) / 2
        avg_pupil_rel_y = (left_pupil_rel_y + right_pupil_rel_y) / 2
        
        # Determine gaze direction
        # Apply Gaussian blur simulation for pupil movement analysis
        # (In a real implementation, this would involve more complex eye tracking)
        
        # Use normalized vector
        gaze_x = avg_pupil_rel_x * 100  # Scale for better visibility
        gaze_y = avg_pupil_rel_y * 100
        
        # Return gaze direction as a vector (x, y) where:
        # x > 0: looking right, x < 0: looking left
        # y > 0: looking down, y < 0: looking up
        return (gaze_x, gaze_y)
    
    def estimate_head_pose(self, landmarks, image_shape):
        """Estimate head pose using solvePnP algorithm"""
        if landmarks is None:
            return None
        
        # 3D model points (simplified)
        model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),   # Right eye right corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ])
        
        # 2D image points from landmarks
        image_points = np.array([
            (landmarks[1].x * image_shape[1], landmarks[1].y * image_shape[0]),    # Nose tip
            (landmarks[199].x * image_shape[1], landmarks[199].y * image_shape[0]), # Chin
            (landmarks[33].x * image_shape[1], landmarks[33].y * image_shape[0]),   # Left eye left corner
            (landmarks[263].x * image_shape[1], landmarks[263].y * image_shape[0]), # Right eye right corner
            (landmarks[61].x * image_shape[1], landmarks[61].y * image_shape[0]),   # Left mouth corner
            (landmarks[291].x * image_shape[1], landmarks[291].y * image_shape[0])  # Right mouth corner
        ], dtype="double")
        
        # Camera matrix estimation
        focal_length = image_shape[1]
        center = (image_shape[1]/2, image_shape[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        
        # Assuming no lens distortion
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs)
        
        if not success:
            return None
            
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Calculate Euler angles
        euler_angles = self.rotation_matrix_to_euler_angles(rotation_matrix)
        
        # Convert to degrees
        euler_angles = np.degrees(euler_angles)
        
        # Return yaw, pitch, roll in degrees
        return euler_angles
    
    def rotation_matrix_to_euler_angles(self, R):
        """Convert rotation matrix to Euler angles"""
        # Calculate Euler angles
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        if sy > 1e-6:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
            
        return np.array([x, y, z])
    
    def establish_baseline(self, num_frames=30):
        """Establish baseline for gaze direction and head pose from initial frames"""
        print("Establishing eye and head position baseline...")
        gaze_directions = []
        head_poses = []
        
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image with FaceMesh
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                
                # Get gaze direction
                gaze = self.get_gaze_direction(face_landmarks)
                if gaze:
                    gaze_directions.append(gaze)
                
                # Get head pose
                pose = self.estimate_head_pose(face_landmarks, frame.shape)
                if pose is not None:
                    head_poses.append(pose)
        
        # Set gaze direction baseline (average of detected positions)
        if gaze_directions:
            avg_gaze_x = sum(gaze[0] for gaze in gaze_directions) / len(gaze_directions)
            avg_gaze_y = sum(gaze[1] for gaze in gaze_directions) / len(gaze_directions)
            self.gaze_direction_baseline = (avg_gaze_x, avg_gaze_y)
            print(f"Baseline gaze direction: {self.gaze_direction_baseline}")
        else:
            print("Warning: Could not establish gaze direction baseline.")
            
        # Set head pose baseline (average of detected angles)
        if head_poses:
            avg_yaw = sum(pose[1] for pose in head_poses) / len(head_poses)
            avg_pitch = sum(pose[0] for pose in head_poses) / len(head_poses)
            avg_roll = sum(pose[2] for pose in head_poses) / len(head_poses)
            self.head_pose_baseline = (avg_yaw, avg_pitch, avg_roll)
            print(f"Baseline head pose (yaw, pitch, roll): {self.head_pose_baseline}")
        else:
            print("Warning: Could not establish head pose baseline.")
            
        # Reset video position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def detect_looking_away(self, rgb_frame):
        """Detect if person is looking away from the screen using eye gaze and head pose"""
        if not self.gaze_direction_baseline or not self.head_pose_baseline:
            return False
            
        # Process the image with FaceMesh
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return False  # No face detected
            
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # Get current gaze direction
        current_gaze = self.get_gaze_direction(face_landmarks)
        
        # Get current head pose
        current_pose = self.estimate_head_pose(face_landmarks, (rgb_frame.shape[0], rgb_frame.shape[1]))
        
        if current_gaze is None or current_pose is None:
            return False
            
        # Calculate deviation from baseline for gaze
        gaze_x_dev = abs(current_gaze[0] - self.gaze_direction_baseline[0])
        gaze_y_dev = abs(current_gaze[1] - self.gaze_direction_baseline[1])
        
        # Calculate deviation from baseline for head pose
        yaw_dev = abs(current_pose[1] - self.head_pose_baseline[0])  # Left-right head rotation
        pitch_dev = abs(current_pose[0] - self.head_pose_baseline[1])  # Up-down head rotation
        
        # Define thresholds for gaze and head movement
        GAZE_THRESHOLD = 5.0  # Threshold for significant gaze movement
        YAW_THRESHOLD = 15.0  # Threshold for significant horizontal head rotation (degrees)
        PITCH_THRESHOLD = 15.0  # Threshold for significant vertical head rotation (degrees)
        
        # Detect looking away based on significant gaze or head pose deviation
        looking_left_right = gaze_x_dev > GAZE_THRESHOLD or yaw_dev > YAW_THRESHOLD
        looking_up_down = gaze_y_dev > GAZE_THRESHOLD or pitch_dev > PITCH_THRESHOLD
        
        return looking_left_right or looking_up_down
    
    def detect_person_count_violations(self, results):
        """
        Detect if a person is missing or if there are additional people
        
        Returns:
            dict: Contains two boolean values:
                  - "person_missing": True if no person detected
                  - "additional_person": True if more than one person detected
        """
        # Check if there are any detections at all
        if len(results[0].boxes) == 0:
            return {"person_missing": True, "additional_person": False}
        
        # Count the number of persons detected
        persons = [detection for detection in results[0].boxes.data.tolist() 
                  if detection[5] == 0]  # class 0 is person
        
        person_count = len(persons)
        
        return {
            "person_missing": person_count == 0,  # Missing if zero people
            "additional_person": person_count > self.expected_person_count  # Additional if more than expected (1)
        }
    
    def process_video(self):
        """Process the video and detect cheating behaviors"""
        print("Processing video for cheating detection...")
        
        # Establish baseline from first few seconds
        self.establish_baseline()
        
        frame_count = 0
        start_time = time.time()
        
        # CUDA optimization - set inference parameters
        cuda_params = {
            'verbose': False,
            'device': self.device,
            'conf': 0.5,  # Confidence threshold
            'iou': 0.45,  # IoU threshold for NMS
            'batch': self.batch_size  # Batch size parameter
        }
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Only process every nth frame for efficiency
            if frame_count % self.frame_skip != 0:
                frame_count += 1
                continue
                
            # Calculate current timestamp
            timestamp = frame_count / self.fps
            timestamp_str = str(timedelta(seconds=int(timestamp)))
            
            # Object detection (phones, books, people)
            object_results = self.object_detector(frame, **cuda_params)
            
            # Check for person count violations (missing or additional)
            person_violations = self.detect_person_count_violations(object_results)
            
            if person_violations["person_missing"]:
                self.violations["person_missing"].append(timestamp)
                print(f"Person missing from frame at {timestamp_str}")
                
            if person_violations["additional_person"]:
                self.violations["additional_person"].append(timestamp)
                print(f"Additional person detected at {timestamp_str}")
            
            # Check for prohibited objects
            for detection in object_results[0].boxes.data.tolist():
                class_id = int(detection[5])
                confidence = detection[4]
                
                if class_id in self.target_classes and confidence > 0.5:
                    object_type = self.target_classes[class_id]
                    
                    if object_type == "cell phone":
                        self.violations["cell_phone"].append(timestamp)
                        print(f"Cell phone detected at {timestamp_str}")
                    
                    elif object_type == "book":
                        self.violations["book"].append(timestamp)
                        print(f"Book detected at {timestamp_str}")
            
            # Face orientation detection (less frequent check)
            if frame_count % self.face_detection_interval == 0:
                # Only check for looking away if at least one person is present
                if not person_violations["person_missing"]:
                    # Convert the BGR image to RGB for MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    if self.detect_looking_away(rgb_frame):
                        self.violations["looking_away"].append(timestamp)
                        print(f"Looking away detected at {timestamp_str}")
            
            # Progress update
            if frame_count % (self.fps * 10) == 0:  # Every 10 seconds of video
                progress = (frame_count / self.total_frames) * 100
                elapsed = time.time() - start_time
                remaining = (elapsed / frame_count) * (self.total_frames - frame_count) if frame_count > 0 else 0
                print(f"Progress: {progress:.1f}% | Time remaining: {remaining:.1f} seconds")
            
            frame_count += 1
            
            # Clear CUDA cache periodically to prevent memory overflow
            if self.device == 'cuda' and frame_count % (self.fps * 60) == 0:  # Every minute of video
                torch.cuda.empty_cache()

        # Record total processing time
        self.processing_time = time.time() - start_time
        print(f"Processing completed in {self.processing_time:.2f} seconds")
        
        # Clean up
        self.cap.release()
        self.face_mesh.close()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        print("Video processing complete.")
    
    def consolidate_violations(self, time_threshold=3.0):
        """
        Consolidate violation timestamps that are close to each other
        
        Args:
            time_threshold (float): Threshold in seconds to consolidate nearby detections
        """
        for violation_type in self.violations:
            timestamps = sorted(self.violations[violation_type])
            if not timestamps:
                continue
                
            consolidated = [timestamps[0]]
            
            for timestamp in timestamps[1:]:
                if timestamp - consolidated[-1] > time_threshold:
                    consolidated.append(timestamp)
            
            self.violations[violation_type] = consolidated
    
    def save_results(self):
        """Save detection results to a file"""
        self.consolidate_violations()
        
        # Generate output filename based on input video
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_file = os.path.join(self.output_dir, f"{video_name}_violations.txt")
        
        with open(output_file, 'w') as f:
            f.write(f"Cheating Detection Results for: {self.video_path}\n")
            f.write(f"Video duration: {self.video_duration:.2f} seconds\n\n")
            f.write(f"Processing time: {self.processing_time:.2f} seconds\n\n")
            
            violation_count = sum(len(timestamps) for timestamps in self.violations.values())
            f.write(f"Total violations detected: {violation_count}\n\n")
            
            for violation_type, timestamps in self.violations.items():
                if timestamps:
                    f.write(f"{violation_type.replace('_', ' ').title()} Violations ({len(timestamps)}):\n")
                    for ts in timestamps:
                        f.write(f"  {str(timedelta(seconds=int(ts)))}\n")
                    f.write("\n")
            
        print(f"Results saved to {output_file}")
        
        # Return summary of violations
        return {vtype: len(timestamps) for vtype, timestamps in self.violations.items()}

def main():
    """Main function to run the cheating detection system"""
    parser = argparse.ArgumentParser(description='Detect cheating behaviors in exam videos')
    parser.add_argument('video', help='Path to the input video file')
    parser.add_argument('--output', default='results', help='Directory to save results')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage even if CUDA is available')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference (larger values use more VRAM)')
    parser.add_argument('--half', action='store_true', help='Use FP16 half-precision inference for faster processing')
    args = parser.parse_args()
    
    # Check CUDA availability
    if not args.cpu and torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        if args.half:
            print("Using FP16 half-precision for faster inference")
    else:
        if not torch.cuda.is_available() and not args.cpu:
            print("CUDA is not available. Falling back to CPU.")
        else:
            print("Using CPU as requested.")
    
    # Set environment variables for better CUDA performance
    if not args.cpu and torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
    
    try:
        # Configure PyTorch to use TensorFloat32 on Ampere or newer GPUs for better performance
        if not args.cpu and torch.cuda.is_available():
            # Check if GPU supports TF32 (Ampere or newer)
            compute_capability = torch.cuda.get_device_capability(0)
            if compute_capability[0] >= 8:  # Ampere and newer have compute capability >= 8.0
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("Enabled TF32 for faster computation on Ampere+ GPUs")
        
        # Create detector instance with parameters
        detector = ExamCheatingDetector(
            args.video, 
            args.output, 
            use_cuda=not args.cpu,
            half_precision=args.half,
            batch_size=args.batch_size
        )
        
        detector.process_video()
        violation_summary = detector.save_results()
        
        print("\nCheating Detection Summary:")
        for vtype, count in violation_summary.items():
            print(f"  {vtype.replace('_', ' ').title()}: {count}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()