#MAKE FastAPI of this.
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import math
import time
from datetime import timedelta

# --- MediaPipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Helper Function for Gaze ---
def get_gaze_direction_refined(image_width, image_height, face_landmarks_normalized, current_gaze_threshold):
    gaze_text = "GAZE: N/A"
    gaze_color = (0, 165, 255)  # Orange for N/A or undetermined
    sum_gaze_ratios = 0.0
    valid_eyes_count = 0

    left_iris_indices = list(range(473, 478)) # landmarks for left iris
    right_iris_indices = list(range(468, 473)) # landmarks for right iris

    left_eye_outer_corner_idx = 33
    left_eye_inner_corner_idx = 133
    right_eye_outer_corner_idx = 263 # Subject's right eye, temple side
    right_eye_inner_corner_idx = 362 # Subject's right eye, nose side

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
    except Exception: pass

    # --- Process Right Eye ---
    try:
        right_iris_coords_x = [face_landmarks_normalized[i].x * image_width for i in right_iris_indices]
        right_pupil_x = np.mean(right_iris_coords_x)
        right_outer_x = face_landmarks_normalized[right_eye_outer_corner_idx].x * image_width
        right_inner_x = face_landmarks_normalized[right_eye_inner_corner_idx].x * image_width

        actual_right_eye_inner_x = min(right_outer_x, right_inner_x) # Nose side
        actual_right_eye_outer_x = max(right_outer_x, right_inner_x) # Temple side
        eye_width_right = abs(actual_right_eye_outer_x - actual_right_eye_inner_x)

        if eye_width_right > 5:
            eye_center_x_right = (actual_right_eye_inner_x + actual_right_eye_outer_x) / 2
            gaze_ratio_right = (right_pupil_x - eye_center_x_right) / (eye_width_right / 2 + 1e-6)
            sum_gaze_ratios += gaze_ratio_right
            valid_eyes_count += 1
    except Exception: pass

    final_gaze_ratio = 0.0
    if valid_eyes_count > 0:
        final_gaze_ratio = sum_gaze_ratios / valid_eyes_count
        if final_gaze_ratio < -current_gaze_threshold: # Negative ratio -> looking left
            gaze_text = "GAZE: LEFT"
            gaze_color = (0, 0, 255)  # Red
        elif final_gaze_ratio > current_gaze_threshold: # Positive ratio -> looking right
            gaze_text = "GAZE: RIGHT"
            gaze_color = (0, 0, 255)  # Red
        else:
            gaze_text = "GAZE: CENTER"
            gaze_color = (0, 255, 0)  # Green
        return gaze_text, gaze_color, final_gaze_ratio

    return "GAZE: N/A", (0, 165, 255), 0.0


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Gaze Detection Demo")
st.title("Gaze Detection System Demo")
st.markdown("""
Upload a video. The system detects extreme gaze (looking left/right).
Alerts are checked at defined intervals. Flagged frames and a timestamped log will be available.
""")

st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'mov', 'avi', 'mkv'])

st.sidebar.subheader("Detection Settings")
gaze_threshold_slider = st.sidebar.slider("Gaze Deviation Threshold", 0.05, 0.8, 0.35, 0.01,
                                          help="Higher value = less sensitive to gaze deviation.")
alert_interval_seconds = st.sidebar.slider("Alert Check Interval (seconds)", 0.5, 10.0, 2.0, 0.5,
                                           help="How often to evaluate for an alert condition.")
max_alert_frames_display = st.sidebar.slider("Max Alert Frames to Display", 5, 50, 10, 1,
                                             help="Max flagged frames in dashboard.")

min_detection_confidence = st.sidebar.slider("Min Face Detection Confidence", 0.1, 1.0, 0.5, 0.05)
min_tracking_confidence = st.sidebar.slider("Min Landmark Tracking Confidence", 0.1, 1.0, 0.5, 0.05)

st.sidebar.subheader("CUDA & Performance")
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    st.sidebar.success(f"OpenCV CUDA available on {cv2.cuda.getCudaEnabledDeviceCount()} device(s).")
else:
    st.sidebar.warning("OpenCV CUDA not available/enabled.")

if 'alert_log' not in st.session_state: st.session_state.alert_log = []
if 'alert_frames_data' not in st.session_state: st.session_state.alert_frames_data = []


# --- Main Processing Logic ---
if uploaded_file is not None:
    st.session_state.alert_log = []
    st.session_state.alert_frames_data = []
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error: Could not open video file.")
    else:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.sidebar.markdown(f"---")
        st.sidebar.markdown(f"**Video Info:** Res: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, FPS: {video_fps:.2f}, Total Frames: {video_total_frames if video_total_frames > 0 else 'N/A'}")

        live_feed_col, stats_col = st.columns([3, 1.5])
        with live_feed_col:
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
        with stats_col:
            stats_placeholder = st.empty()
        
        st.markdown("---")
        st.subheader("Flagged Gaze Event Frames")
        alert_frames_placeholder = st.empty()

        with mp_face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True, # refine_landmarks is crucial for iris
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence) as face_mesh:

            stop_button = st.button("Stop Processing and Show Results")
            frame_idx = 0
            processing_start_time = time.time()
            last_alert_check_time = time.time()
            gaze_alert_count = 0
            gaze_status_at_last_check = "GAZE: N/A"
            raw_gaze_ratio_for_display = 0.0
            current_gaze_status_text = "GAZE: N/A"
            current_gaze_status_color = (0,165,255)

            while cap.isOpened() and not stop_button:
                success, frame = cap.read()
                if not success: break
                
                current_process_time = time.time()
                current_video_time_sec = frame_idx / video_fps if video_fps > 0 else 0
                
                original_frame_for_alert = frame.copy()
                output_image_display = cv2.flip(frame.copy(), 1)
                image_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_input.flags.writeable = False
                results = face_mesh.process(image_input)

                if results.multi_face_landmarks:
                    for face_landmarks_obj in results.multi_face_landmarks:
                        face_landmarks_list = face_landmarks_obj.landmark
                        
                        temp_rgb_flipped = cv2.cvtColor(output_image_display, cv2.COLOR_BGR2RGB)
                        # Only draw face mesh essentials, iris is important for gaze context
                        mp_drawing.draw_landmarks(image=temp_rgb_flipped, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(image=temp_rgb_flipped, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                        output_image_display = cv2.cvtColor(temp_rgb_flipped, cv2.COLOR_RGB2BGR)

                        img_h, img_w = frame.shape[:2]
                        current_gaze_status_text, current_gaze_status_color, raw_gaze_ratio_for_display = get_gaze_direction_refined(img_w, img_h, face_landmarks_list, gaze_threshold_slider)
                else:
                    current_gaze_status_text, current_gaze_status_color, raw_gaze_ratio_for_display = "GAZE: NO FACE", (0,165,255), 0.0

                if (current_process_time - last_alert_check_time) >= alert_interval_seconds:
                    last_alert_check_time = current_process_time
                    alert_this_interval, alert_type_this_interval = False, ""
                    if "LEFT" in current_gaze_status_text or "RIGHT" in current_gaze_status_text:
                        gaze_alert_count += 1; gaze_status_at_last_check = current_gaze_status_text; alert_this_interval=True; alert_type_this_interval=f"{current_gaze_status_text}"
                    else: gaze_status_at_last_check = "GAZE: CENTER (interval)"
                    
                    if alert_this_interval:
                        timestamp_str = str(timedelta(seconds=int(current_video_time_sec)))
                        log_entry = f"Timestamp: {timestamp_str}, Event: {alert_type_this_interval.strip()}, Frame_idx: {frame_idx}, GazeRatio: {raw_gaze_ratio_for_display:.2f}"
                        st.session_state.alert_log.append(log_entry)
                        if len(st.session_state.alert_frames_data) < max_alert_frames_display:
                            st.session_state.alert_frames_data.append({"frame_rgb": cv2.cvtColor(original_frame_for_alert, cv2.COLOR_BGR2RGB), "caption": f"{timestamp_str} - {alert_type_this_interval.strip()} (R:{raw_gaze_ratio_for_display:.2f})"})
                
                y_offset = 20; cv2.putText(output_image_display, f"{current_gaze_status_text} (R:{raw_gaze_ratio_for_display:.2f})", (5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_gaze_status_color, 1)
                y_offset += 20; cv2.putText(output_image_display, f"Time: {str(timedelta(seconds=int(current_video_time_sec)))}", (5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                frame_placeholder.image(output_image_display, channels="BGR", use_column_width=True)
                if video_total_frames > 0: progress_bar.progress(frame_idx / video_total_frames)
                
                elapsed_processing_time = time.time() - processing_start_time
                processing_fps = (frame_idx + 1) / elapsed_processing_time if elapsed_processing_time > 0 else 0
                stats_placeholder.markdown(f"""**Processing:** F {frame_idx+1}/{video_total_frames if video_total_frames > 0 else 'N/A'} | FPS: {processing_fps:.2f}
                **Gaze Alerts ({alert_interval_seconds}s):** Status: <span style='color:{"red" if "LEFT" in gaze_status_at_last_check or "RIGHT" in gaze_status_at_last_check else "green"};'>{gaze_status_at_last_check.replace("GAZE: ","")}</span> ({gaze_alert_count})
                """, unsafe_allow_html=True)
                frame_idx += 1

            cap.release(); tfile.close(); progress_bar.progress(1.0); st.success("Video processing complete!")
            if st.session_state.alert_log:
                log_content = "\n".join(st.session_state.alert_log)
                st.download_button(label="Download Gaze Alert Log (TXT)", data=log_content, file_name="gaze_alert_log.txt", mime="text/plain")
            else: st.info("No gaze alerts logged.")
            if st.session_state.alert_frames_data:
                alert_frames_placeholder.empty()
                num_cols = st.sidebar.number_input("Alert Frame Columns", 1, 5, 3)
                cols = alert_frames_placeholder.columns(num_cols)
                for i, alert_data in enumerate(st.session_state.alert_frames_data):
                    cols[i % num_cols].image(alert_data["frame_rgb"], caption=alert_data["caption"], use_column_width=True)
            else: alert_frames_placeholder.info("No gaze alert frames to display.")
            if gaze_alert_count > 0 : st.balloons()
else:
    st.info("Upload a video file to start processing.")
    if st.session_state.alert_log:
        st.subheader("Previous Run Gaze Alert Log")
        st.text_area("Log", "\n".join(st.session_state.alert_log), height=150)
    if st.session_state.alert_frames_data:
        st.subheader("Previous Run Flagged Gaze Frames")
        num_cols_prev = st.sidebar.number_input("Prev. Alert Frame Columns", 1, 5, 3, key="prev_cols")
        cols_prev = st.columns(num_cols_prev)
        for i, alert_data in enumerate(st.session_state.alert_frames_data):
            if i >= max_alert_frames_display: break 
            cols_prev[i % num_cols_prev].image(alert_data["frame_rgb"], caption=alert_data["caption"], use_column_width=True)