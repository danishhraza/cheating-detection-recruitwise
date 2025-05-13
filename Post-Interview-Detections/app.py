#remove face detection from this. face detection happening in app2.py. MAKE FastAPI of this. 
import streamlit as st
import cv2
import numpy as np
import torch
import time
import os
import tempfile
from datetime import timedelta
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import io

# Patch for PyTorch compatibility with Streamlit
# This prevents Streamlit from trying to access torch._classes.__path__._path
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

# Set page configuration
st.set_page_config(
    page_title="Exam Cheating Detection Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stProgress .st-eb {
        background-color: #1E88E5;
    }
    .highlight {
        background-color: #ffeb3b;
        padding: 0.2rem;
        border-radius: 0.2rem;
    }
    .violation-card {
        background-color: #f8f9fa;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

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
        st.info(f"Loading detection models on {self.device}...")
        
        # Load object detector - Note: we'll handle half-precision differently
        with st.spinner("Loading object detection model (YOLOv8x)..."):
            self.object_detector = YOLO("yolov8x.pt")
            self.object_detector.to(self.device)
            # We won't use .half() directly as it causes type mismatch errors
            
        # Load face detector    
        with st.spinner("Loading face detection model (YOLOv8n)..."):
            self.face_detector = YOLO("yolov8n.pt")
            self.face_detector.to(self.device)
            # We won't use .half() directly as it causes type mismatch errors
        
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
            st.error(f"Error: Could not open video file {video_path}")
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

        # Save frames with violations
        self.violation_frames = {
            "cell_phone": [],
            "additional_person": [],
            "book": [],
            "looking_away": [],
            "person_missing": []
        }

        # Time tracking
        self.processing_time = 0
        
        st.success(f"Video loaded: {self.total_frames} frames, {self.video_duration:.2f} seconds")
        st.info(f"Using device: {self.device}")
        if self.device == 'cuda':
            st.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            st.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def establish_baseline(self, num_frames=30):
        """Establish baseline for face position from initial frames"""
        st.info("Establishing face position baseline...")
        face_positions = []
        
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Detect face position
            # Configure half-precision via params rather than model.half()
            face_detect_params = {
                'verbose': False,
                'device': self.device,
                'half': self.half_precision  # Use half parameter instead of model.half()
            }
            
            face_results = self.face_detector(frame, **face_detect_params)
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
            st.info(f"Baseline face position: {self.face_position_baseline}")
        else:
            st.warning("Warning: Could not establish face position baseline. No faces detected in initial frames.")
            
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
    
    def process_video(self, progress_bar, status_text):
        """Process the video and detect cheating behaviors with UI updates"""
        st.info("Processing video for cheating detection...")
        
        # Establish baseline from first few seconds
        self.establish_baseline()
        
        frame_count = 0
        start_time = time.time()
        
        # CUDA optimization - set inference parameters
        # Configure half-precision via params rather than model.half()
        cuda_params = {
            'verbose': False,
            'device': self.device,
            'conf': 0.5,  # Confidence threshold
            'iou': 0.45,  # IoU threshold for NMS
            'batch': self.batch_size,  # Batch size parameter
            'half': self.half_precision  # Use half parameter instead of model.half()
        }

        # Create a placeholder for displaying the current frame
        frame_placeholder = st.empty()
        
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
            
            # Update status text
            status_text.text(f"Processing frame {frame_count}/{self.total_frames} at {timestamp_str}")
            
            # Object detection (phones, books, people)
            object_results = self.object_detector(frame, **cuda_params)
            
            # Check for person count violations (missing or additional)
            person_violations = self.detect_person_count_violations(object_results)
            
            violations_in_frame = False
            
            if person_violations["person_missing"]:
                self.violations["person_missing"].append(timestamp)
                status_text.warning(f"Person missing from frame at {timestamp_str}")
                violations_in_frame = True
                # Save frame with violation
                self.violation_frames["person_missing"].append((timestamp, frame.copy()))
                
            if person_violations["additional_person"]:
                self.violations["additional_person"].append(timestamp)
                status_text.warning(f"Additional person detected at {timestamp_str}")
                violations_in_frame = True
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
                        status_text.warning(f"Cell phone detected at {timestamp_str}")
                        violations_in_frame = True
                        # Save frame with violation with bounding box
                        annotated_frame = frame.copy()
                        x1, y1, x2, y2 = map(int, detection[:4])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(annotated_frame, "Cell Phone", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        self.violation_frames["cell_phone"].append((timestamp, annotated_frame))
                    
                    elif object_type == "book":
                        self.violations["book"].append(timestamp)
                        status_text.warning(f"Book detected at {timestamp_str}")
                        violations_in_frame = True
                        # Save frame with violation with bounding box
                        annotated_frame = frame.copy()
                        x1, y1, x2, y2 = map(int, detection[:4])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(annotated_frame, "Book", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        self.violation_frames["book"].append((timestamp, annotated_frame))
            
            # Face orientation detection (less frequent check)
            if frame_count % self.face_detection_interval == 0:
                # Only check for looking away if at least one person is present
                if not person_violations["person_missing"]:
                    face_results = self.face_detector(frame, **cuda_params)
                    if self.detect_looking_away(face_results, frame.shape):
                        self.violations["looking_away"].append(timestamp)
                        status_text.warning(f"Looking away detected at {timestamp_str}")
                        violations_in_frame = True
                        # Save frame with violation
                        if len(face_results[0].boxes) > 0:
                            annotated_frame = frame.copy()
                            face_boxes = face_results[0].boxes.data.tolist()
                            face_areas = [(box[2]-box[0])*(box[3]-box[1]) for box in face_boxes]
                            largest_face_idx = face_areas.index(max(face_areas))
                            face_box = face_boxes[largest_face_idx]
                            x1, y1, x2, y2 = map(int, face_box[:4])
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(annotated_frame, "Looking Away", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                            self.violation_frames["looking_away"].append((timestamp, annotated_frame))
            
            # Display the current frame if it has violations
            if violations_in_frame:
                # Convert BGR to RGB for display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, caption=f"Frame at {timestamp_str}", use_container_width=True)
            
            # Progress update
            if frame_count % (self.fps * 5) == 0 or frame_count == self.total_frames - 1:  # Every 5 seconds of video
                progress = (frame_count / self.total_frames)
                progress_bar.progress(progress)
            
            frame_count += 1
            
            # Clear CUDA cache periodically to prevent memory overflow
            if self.device == 'cuda' and frame_count % (self.fps * 60) == 0:  # Every minute of video
                torch.cuda.empty_cache()

        # Record total processing time
        self.processing_time = time.time() - start_time
        status_text.success(f"Processing completed in {self.processing_time:.2f} seconds")
        
        # Clean up
        self.cap.release()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        # Set progress to 100%
        progress_bar.progress(1.0)
        
        status_text.success("Video processing complete.")
    
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
            "looking_away": [],
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
        """Save detection results and return summary"""
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
        
        # Return summary of violations with frames
        return {
            "counts": {vtype: len(timestamps) for vtype, timestamps in self.violations.items()},
            "timestamps": self.violations,
            "frames": self.violation_frames
        }


def main():
    st.markdown("<h1 class='main-header'>Exam Cheating Detection Dashboard</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    This dashboard allows you to upload exam videos and automatically detect potential cheating behaviors including:
    <ul>
        <li>Cell phone usage</li>
        <li>Additional persons in frame</li>
        <li>Book/reference material detection</li>
        <li>Looking away behavior</li>
        <li>Person missing from frame</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.markdown("## Settings")
    
    # CUDA settings
    use_cuda = st.sidebar.checkbox("Use CUDA GPU (if available)", value=True)
    cuda_available = torch.cuda.is_available()
    
    if use_cuda and not cuda_available:
        st.sidebar.warning("CUDA is not available on this system. Using CPU instead.")
        use_cuda = False
    elif use_cuda and cuda_available:
        st.sidebar.success(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Performance settings
    half_precision = st.sidebar.checkbox("Use FP16 half-precision", value=False,
                                       help="Faster processing with slightly lower accuracy")
    
    batch_size = st.sidebar.slider("Batch size", min_value=1, max_value=8, value=1,
                                  help="Higher values use more GPU memory but may be faster")
    
    # Upload video
    uploaded_file = st.file_uploader("Upload exam video", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        # Create temp file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"Video uploaded: {uploaded_file.name}")
        
        # Video info
        video = cv2.VideoCapture(temp_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video.release()
        
        # Display video info
        col1, col2, col3 = st.columns(3)
        col1.metric("Duration", f"{duration:.1f} seconds")
        col2.metric("Resolution", f"{width}x{height}")
        col3.metric("FPS", f"{fps:.1f}")
        
        # Display video preview
        st.video(temp_path)
        
        # Process video button
        if st.button("Detect Cheating Behaviors"):
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create results directory
            os.makedirs("results", exist_ok=True)
            
            try:
                # Initialize detector
                detector = ExamCheatingDetector(
                    temp_path,
                    output_dir="results",
                    use_cuda=use_cuda and cuda_available,
                    half_precision=half_precision,
                    batch_size=batch_size
                )
                
                # Process video
                detector.process_video(progress_bar, status_text)
                
                # Get results
                results = detector.save_results()
                
                # Display results
                st.markdown("<h2 class='sub-header'>Detection Results</h2>", unsafe_allow_html=True)
                
                # Create summary metrics
                summary_cols = st.columns(5)
                violation_types = ["Cell Phone", "Additional Person", "Book", "Looking Away", "Person Missing"]
                violation_keys = ["cell_phone", "additional_person", "book", "looking_away", "person_missing"]
                
                for i, (col, v_type, v_key) in enumerate(zip(summary_cols, violation_types, violation_keys)):
                    col.metric(v_type, results["counts"][v_key])
                
                # Create tabs for different violation types
                tabs = st.tabs(violation_types)
                
                for i, (tab, v_type, v_key) in enumerate(zip(tabs, violation_types, violation_keys)):
                    with tab:
                        if results["counts"][v_key] > 0:
                            st.markdown(f"### {v_type} Violations: {results['counts'][v_key]}")
                            
                            # Display timestamps
                            st.markdown("#### Timestamps:")
                            for ts in results["timestamps"][v_key]:
                                st.markdown(f"<div class='violation-card'>🕒 {str(timedelta(seconds=int(ts)))}</div>", unsafe_allow_html=True)
                            
                            # Display frames if available
                            if results["frames"][v_key]:
                                st.markdown("#### Evidence Frames:")
                                frame_cols = st.columns(min(3, len(results["frames"][v_key])))
                                
                                for j, (ts, frame) in enumerate(results["frames"][v_key]):
                                    col_idx = j % 3
                                    # Convert BGR to RGB
                                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    frame_cols[col_idx].image(rgb_frame, caption=f"At {str(timedelta(seconds=int(ts)))}", use_container_width=True)
                        else:
                            st.info(f"No {v_type.lower()} violations detected")
                
                # Display a chart of violations over time
                st.markdown("<h2 class='sub-header'>Violations Timeline</h2>", unsafe_allow_html=True)
                
                # Prepare data for timeline chart
                timeline_data = []
                
                for v_key, v_type in zip(violation_keys, violation_types):
                    for ts in results["timestamps"][v_key]:
                        timeline_data.append({
                            "Timestamp": int(ts),
                            "Violation Type": v_type
                        })
                
                if timeline_data:
                    timeline_df = pd.DataFrame(timeline_data)
                    
                    # Create a chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot each violation type with a different color
                    for i, v_type in enumerate(violation_types):
                        if v_type in timeline_df["Violation Type"].values:
                            v_data = timeline_df[timeline_df["Violation Type"] == v_type]
                            ax.scatter(v_data["Timestamp"], [i] * len(v_data), label=v_type, s=100)
                    
                    ax.set_yticks(range(len(violation_types)))
                    ax.set_yticklabels(violation_types)
                    ax.set_xlabel("Time (seconds)")
                    ax.set_title("Violations Timeline")
                    ax.grid(axis='x', linestyle='--', alpha=0.7)
                    
                    # Format x-axis to show time in minutes:seconds
                    def format_time(x, pos):
                        return str(timedelta(seconds=int(x)))
                    
                    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No violations detected to display in timeline")
                
                # Provide download link for results
                with open(os.path.join("results", f"{os.path.splitext(uploaded_file.name)[0]}_violations.txt"), "r") as f:
                    report_content = f.read()
                
                st.download_button(
                    label="Download Full Report",
                    data=report_content,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_violations.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    else:
        # Display placeholder message when no file is uploaded
        st.info("Please upload a video file to begin analysis")

if __name__ == "__main__":
    main()