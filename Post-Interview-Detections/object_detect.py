import cv2
import numpy as np
import torch
import time
import os
from datetime import timedelta
from ultralytics import YOLO

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
        print("Loading object detection model (YOLOv8x)...")
        self.object_detector = YOLO("yolov8x.pt")
        self.object_detector.to(self.device)
        
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
        
        # We assume exactly 1 person should be in the frame by default
        self.expected_person_count = 1
        
        # Violation tracking
        self.violations = {
            "cell_phone": [],
            "additional_person": [],
            "book": [],
            "person_missing": []
        }

        # Save frames with violations
        self.violation_frames = {
            "cell_phone": [],
            "additional_person": [],
            "book": [],
            "person_missing": []
        }

        # Time tracking
        self.processing_time = 0
        
        print(f"Video loaded: {self.total_frames} frames, {self.video_duration:.2f} seconds")
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
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
        
        frame_count = 0
        start_time = time.time()
        
        # CUDA optimization - set inference parameters
        cuda_params = {
            'verbose': False,
            'device': self.device,
            'conf': 0.5,  # Confidence threshold
            'iou': 0.45,  # IoU threshold for NMS
            'batch': self.batch_size,  # Batch size parameter
            'half': self.half_precision  # Use half parameter instead of model.half()
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
                # Save frame with violation
                self.violation_frames["person_missing"].append((timestamp, frame.copy()))
                
            if person_violations["additional_person"]:
                self.violations["additional_person"].append(timestamp)
                print(f"Additional person detected at {timestamp_str}")
                # Save frame with violation
                self.violation_frames["additional_person"].append((timestamp, frame.copy()))
            
            # Check for prohibited objects
            for detection in object_results[0].boxes.data.tolist():
                class_id = int(detection[5])
                confidence = detection[4]
                
                if class_id in self.target_classes and confidence > 0.5:
                    object_type = self.target_classes[class_id]
                    
                    if object_type == "cell phone":
                        self.violations["cell_phone"].append(timestamp)
                        print(f"Cell phone detected at {timestamp_str}")
                        # Save frame with violation with bounding box
                        annotated_frame = frame.copy()
                        x1, y1, x2, y2 = map(int, detection[:4])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(annotated_frame, "Cell Phone", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        self.violation_frames["cell_phone"].append((timestamp, annotated_frame))
                    
                    elif object_type == "book":
                        self.violations["book"].append(timestamp)
                        print(f"Book detected at {timestamp_str}")
                        # Save frame with violation with bounding box
                        annotated_frame = frame.copy()
                        x1, y1, x2, y2 = map(int, detection[:4])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(annotated_frame, "Book", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        self.violation_frames["book"].append((timestamp, annotated_frame))
            
            # Progress update
            if frame_count % (self.fps * 5) == 0:  # Every 5 seconds of video
                progress = (frame_count / self.total_frames) * 100
                print(f"Progress: {progress:.1f}%")
            
            frame_count += 1
            
            # Clear CUDA cache periodically to prevent memory overflow
            if self.device == 'cuda' and frame_count % (self.fps * 60) == 0:  # Every minute of video
                torch.cuda.empty_cache()

        # Record total processing time
        self.processing_time = time.time() - start_time
        print(f"Processing completed in {self.processing_time:.2f} seconds")
        
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
        consolidated_frames = {
            "cell_phone": [],
            "additional_person": [],
            "book": [],
            "person_missing": []
        }
        
        for violation_type in self.violations:
            timestamps = sorted(self.violations[violation_type])
            if not timestamps:
                continue
                
            consolidated = [timestamps[0]]
            
            # Get corresponding frames for consolidated timestamps
            if self.violation_frames[violation_type]:
                # Find the frame for the first timestamp
                for ts, frame in self.violation_frames[violation_type]:
                    if abs(ts - timestamps[0]) < 0.5:  # 0.5 second threshold
                        consolidated_frames[violation_type].append((timestamps[0], frame))
                        break
            
            for timestamp in timestamps[1:]:
                if timestamp - consolidated[-1] > time_threshold:
                    consolidated.append(timestamp)
                    
                    # Find corresponding frame for this timestamp
                    if self.violation_frames[violation_type]:
                        for ts, frame in self.violation_frames[violation_type]:
                            if abs(ts - timestamp) < 0.5:  # 0.5 second threshold
                                consolidated_frames[violation_type].append((timestamp, frame))
                                break
            
            self.violations[violation_type] = consolidated
        
        # Update violation_frames with consolidated frames
        self.violation_frames = consolidated_frames
    
    def save_results(self):
        """Save detection results and return summary in required format"""
        self.consolidate_violations()
        
        # Generate structured output with only timestamps
        output = {}
        
        # Add each violation type with its timestamps
        for violation_type, timestamps in self.violations.items():
            output[violation_type] = timestamps
        
        # Output to file in requested format
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_file = os.path.join(self.output_dir, f"{video_name}_violations.txt")
        
        with open(output_file, 'w') as f:
            for violation_type, timestamps in output.items():
                f.write(f"{violation_type} = {timestamps}\n")
        
        # Return the same structured output
        return output


def main():
    # Example usage
    video_path = "exam_video.mp4"  # Replace with your video path
    
    # Initialize detector
    detector = ExamCheatingDetector(
        video_path,
        output_dir="results",
        use_cuda=True,  # Set to False if no GPU available
        half_precision=False,
        batch_size=1
    )
    
    # Process video
    detector.process_video()
    
    # Get and print results
    results = detector.save_results()
    
    # Print summary
    print("\nDetection Summary:")
    for violation_type, count in results["counts"].items():
        print(f"{violation_type.replace('_', ' ').title()}: {count}")

if __name__ == "__main__":
    main()