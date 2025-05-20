import cv2
import numpy as np
import mediapipe as mp
from datetime import timedelta

def get_gaze_direction(image_width, image_height, face_landmarks_normalized, gaze_threshold=0.35):
    """
    Analyze face landmarks to determine gaze direction.
    
    Args:
        image_width (int): Width of the image
        image_height (int): Height of the image
        face_landmarks_normalized (list): MediaPipe normalized face landmarks
        gaze_threshold (float): Threshold to determine extreme gaze (0.05-0.8)
        
    Returns:
        tuple: (gaze_direction_text, is_looking_away, gaze_ratio)
    """
    sum_gaze_ratios = 0.0
    valid_eyes_count = 0

    left_iris_indices = list(range(473, 478))  # landmarks for left iris
    right_iris_indices = list(range(468, 473))  # landmarks for right iris

    left_eye_outer_corner_idx = 33
    left_eye_inner_corner_idx = 133
    right_eye_outer_corner_idx = 263  # Subject's right eye, temple side
    right_eye_inner_corner_idx = 362  # Subject's right eye, nose side

    # --- Process Left Eye ---
    try:
        left_iris_coords_x = [face_landmarks_normalized[i].x * image_width for i in left_iris_indices]
        left_pupil_x = np.mean(left_iris_coords_x)
        left_outer_x = face_landmarks_normalized[left_eye_outer_corner_idx].x * image_width
        left_inner_x = face_landmarks_normalized[left_eye_inner_corner_idx].x * image_width
        
        actual_left_eye_outer_x = min(left_outer_x, left_inner_x)
        actual_left_eye_inner_x = max(left_outer_x, left_inner_x)
        eye_width_left = abs(actual_left_eye_inner_x - actual_left_eye_outer_x)

        if eye_width_left > 5: 
            eye_center_x_left = (actual_left_eye_outer_x + actual_left_eye_inner_x) / 2
            gaze_ratio_left = (left_pupil_x - eye_center_x_left) / (eye_width_left / 2 + 1e-6)
            sum_gaze_ratios += gaze_ratio_left
            valid_eyes_count += 1
    except Exception:
        pass

    # --- Process Right Eye ---
    try:
        right_iris_coords_x = [face_landmarks_normalized[i].x * image_width for i in right_iris_indices]
        right_pupil_x = np.mean(right_iris_coords_x)
        right_outer_x = face_landmarks_normalized[right_eye_outer_corner_idx].x * image_width
        right_inner_x = face_landmarks_normalized[right_eye_inner_corner_idx].x * image_width

        actual_right_eye_inner_x = min(right_outer_x, right_inner_x)  # Nose side
        actual_right_eye_outer_x = max(right_outer_x, right_inner_x)  # Temple side
        eye_width_right = abs(actual_right_eye_outer_x - actual_right_eye_inner_x)

        if eye_width_right > 5:
            eye_center_x_right = (actual_right_eye_inner_x + actual_right_eye_outer_x) / 2
            gaze_ratio_right = (right_pupil_x - eye_center_x_right) / (eye_width_right / 2 + 1e-6)
            sum_gaze_ratios += gaze_ratio_right
            valid_eyes_count += 1
    except Exception:
        pass

    # Calculate final gaze ratio and determine direction
    if valid_eyes_count > 0:
        final_gaze_ratio = sum_gaze_ratios / valid_eyes_count
        if final_gaze_ratio < -gaze_threshold:  # Negative ratio -> looking left
            return "LEFT", True, final_gaze_ratio
        elif final_gaze_ratio > gaze_threshold:  # Positive ratio -> looking right
            return "RIGHT", True, final_gaze_ratio
        else:
            return "CENTER", False, final_gaze_ratio

    return "N/A", False, 0.0

def detect_looking_away_violations(video_path, gaze_threshold=0.35, alert_interval_seconds=2.0, 
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5):
    """
    Process a video and detect when a person is looking away (left or right).
    
    Args:
        video_path (str): Path to the video file
        gaze_threshold (float): Threshold for gaze deviation (0.05-0.8)
        alert_interval_seconds (float): How often to check for alerts
        min_detection_confidence (float): Minimum confidence for face detection
        min_tracking_confidence (float): Minimum confidence for landmark tracking
        
    Returns:
        list: Array of timestamps when person is looking away
    """
    looking_away_violation = []
    
    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video file.")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with mp_face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=True,  # Required for iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence) as face_mesh:
        
        frame_idx = 0
        last_alert_check_time = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            current_video_time_sec = frame_idx / video_fps if video_fps > 0 else 0
            
            # Process frame at specified intervals
            if (current_video_time_sec - last_alert_check_time) >= alert_interval_seconds:
                last_alert_check_time = current_video_time_sec
                
                # Process image with MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = face_mesh.process(image_rgb)
                
                if results.multi_face_landmarks:
                    for face_landmarks_obj in results.multi_face_landmarks:
                        face_landmarks_list = face_landmarks_obj.landmark
                        
                        img_h, img_w = frame.shape[:2]
                        gaze_direction, is_looking_away, gaze_ratio = get_gaze_direction(
                            img_w, img_h, face_landmarks_list, gaze_threshold)
                        
                        if is_looking_away:
                            # Store the timestamp in seconds instead of formatted string
                            looking_away_violation.append({
                                "timestamp": current_video_time_sec,  # Changed to seconds
                                "direction": gaze_direction,
                                "frame_idx": frame_idx,
                                "gaze_ratio": gaze_ratio
                            })
            
            frame_idx += 1
    
    cap.release()
    return looking_away_violation

# Example usage:
# violations = detect_looking_away_violations("path_to_video.mp4")
# print(violations)