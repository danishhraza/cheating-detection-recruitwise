import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
from datetime import timedelta
import argparse
import os

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
            
        # Load face detector    
        self.face_detector = YOLO("yolov8n.pt")
        self.face_detector.to(self.device)
        if self.half_precision and self.device == 'cuda':
            self.face_detector = self.face_detector.half()
        
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
        self.face_position_baseline = None  # Will track initial face position
        
        # Violation tracking
        self.violations = {
            "cell_phone": [],
            "additional_person": [],
            "book": [],
            "looking_away": [],
            "person_missing": []
        }
        
        print(f"Video loaded: {self.total_frames} frames, {self.video_duration:.2f} seconds")
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def establish_baseline(self, num_frames=30):
        """Establish baseline for face position from initial frames"""
        print("Establishing face position baseline...")
        face_positions = []
        
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Detect face position
            face_results = self.face_detector(frame, verbose=False)
            if len(face_results[0].boxes.data) > 0:
                # Get the largest face (assume it's the primary person)
                face_boxes = face_results[0].boxes.data.tolist()
                face_areas = [(box[2]-box[0])*(box[3]-box[1]) for box in face_boxes]
                largest_face_idx = face_areas.index(max(face_areas))
                face_box = face_boxes[largest_face_idx]
                
                # Calculate face center
                face_center_x = (face_box[0] + face_box[2]) / 2
                face_center_y = (face_box[1] + face_box[3]) / 2
                face_positions.append((face_center_x, face_center_y))
        
        # Set face position baseline (average of detected positions)
        if face_positions:
            avg_x = sum(pos[0] for pos in face_positions) / len(face_positions)
            avg_y = sum(pos[1] for pos in face_positions) / len(face_positions)
            self.face_position_baseline = (avg_x, avg_y)
            print(f"Baseline face position: {self.face_position_baseline}")
        else:
            print("Warning: Could not establish face position baseline. No faces detected in initial frames.")
            
        # Reset video position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def detect_looking_away(self, face_results, frame_shape):
        """Detect if person is looking away from the screen"""
        if not self.face_position_baseline:
            return False
            
        if len(face_results[0].boxes.data) == 0:
            return False  # No face detected
            
        # Get the largest face
        face_boxes = face_results[0].boxes.data.tolist()
        face_areas = [(box[2]-box[0])*(box[3]-box[1]) for box in face_boxes]
        largest_face_idx = face_areas.index(max(face_areas))
        face_box = face_boxes[largest_face_idx]
        
        # Calculate face center
        face_center_x = (face_box[0] + face_box[2]) / 2
        face_center_y = (face_box[1] + face_box[3]) / 2
        
        # Calculate deviation from baseline
        baseline_x, baseline_y = self.face_position_baseline
        frame_width, frame_height = frame_shape[1], frame_shape[0]
        
        # Calculate percentage deviation
        x_deviation = abs(face_center_x - baseline_x) / frame_width
        
        # Looking left or right: substantial horizontal deviation
        return x_deviation > 0.15  # 15% deviation threshold
    
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
                    face_results = self.face_detector(frame, **cuda_params)
                    if self.detect_looking_away(face_results, frame.shape):
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
        
        # Clean up
        self.cap.release()
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