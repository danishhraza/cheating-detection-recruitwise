from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import face_recognition
from PIL import Image
import dlib
import json

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Use dlib's face detector for better face detection
face_detector = dlib.get_frontal_face_detector()
# Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
shape_predictor = dlib.shape_predictor(shape_predictor_path)

# JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Configure Flask to use our custom JSON encoder
app.json_encoder = NumpyEncoder

def save_image(image_data, filename):
    """Save image to the uploads folder"""
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image_data.save(image_path)
    return image_path

def align_face(image_path):
    """Align face to standard orientation for better comparison accuracy"""
    try:
        # Load the image
        img = dlib.load_rgb_image(image_path)
        
        # Detect faces
        faces = face_detector(img, 1)
        if len(faces) == 0:
            return None
        
        # Get the first face
        face = faces[0]
        
        # Get facial landmarks
        shape = shape_predictor(img, face)
        
        # Convert landmarks to NumPy array
        landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(shape.num_parts)])
        
        # Calculate eyes center
        left_eye = np.mean(landmarks[36:42], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0)
        
        # Calculate angle for alignment
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Fix: Properly convert center to tuple of integers
        center = (int(img.shape[1] // 2), int(img.shape[0] // 2))
        
        # Rotate to align eyes horizontally
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
        
        # Save aligned image
        aligned_path = image_path.replace('.jpg', '_aligned.jpg')
        cv2.imwrite(aligned_path, cv2.cvtColor(aligned_img, cv2.COLOR_RGB2BGR))
        
        return aligned_path
    except Exception as e:
        print(f"Face alignment error: {str(e)}")
        return None

def preprocess_image(image_path):
    """Preprocess image for better face recognition"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to RGB (face_recognition uses RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect face locations using HOG (more accurate than Haar cascades)
        face_locations = face_recognition.face_locations(img_rgb, model="hog")
        
        if not face_locations:
            return None
        
        # Get the largest face (based on area)
        face_location = max(face_locations, key=lambda rect: (rect[2]-rect[0])*(rect[3]-rect[1]))
        top, right, bottom, left = face_location
        
        # Extract face with margin
        margin = int(0.3 * (bottom - top))
        face_img = img_rgb[
            max(0, top - margin):min(img_rgb.shape[0], bottom + margin),
            max(0, left - margin):min(img_rgb.shape[1], right + margin)
        ]
        
        # Resize to standard size
        face_img = cv2.resize(face_img, (150, 150))
        
        # Apply histogram equalization for better lighting normalization
        face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        face_img_eq = cv2.equalizeHist(face_img_gray)
        face_img = cv2.cvtColor(face_img_eq, cv2.COLOR_GRAY2RGB)
        
        # Save processed face
        processed_path = image_path.replace('.jpg', '_processed.jpg')
        cv2.imwrite(processed_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
        
        return processed_path
    except Exception as e:
        print(f"Image preprocessing error: {str(e)}")
        return None

def calculate_multiple_encodings(image_path, num_encodings=5):
    """Calculate multiple encodings by slightly shifting the face box"""
    try:
        img = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(img, model="hog")
        
        if not face_locations:
            return []
        
        # Get the main face location
        main_location = max(face_locations, key=lambda rect: (rect[2]-rect[0])*(rect[3]-rect[1]))
        top, right, bottom, left = main_location
        
        encodings = []
        
        # Get encoding for main location
        main_encoding = face_recognition.face_encodings(img, [main_location])
        if main_encoding:
            encodings.append(main_encoding[0])
        
        # Create slight variations of the face box
        shifts = [(0, 0), (5, 0), (-5, 0), (0, 5), (0, -5)]
        for dx, dy in shifts[:num_encodings-1]:  # -1 because we already have the main encoding
            shifted_location = (
                max(0, top + dy),
                min(img.shape[1], right + dx),
                min(img.shape[0], bottom + dy),
                max(0, left + dx)
            )
            
            if shifted_location != main_location:  # Skip if we end up with the same box
                encoding = face_recognition.face_encodings(img, [shifted_location])
                if encoding:
                    encodings.append(encoding[0])
        
        return encodings
    except Exception as e:
        print(f"Error calculating encodings: {str(e)}")
        return []

def verify_identity(profile_image_path, live_image_path, threshold=0.5):
    """Advanced face comparison with multiple techniques"""
    try:
        # Step 1: Try to align faces for better comparison
        try:
            aligned_profile = align_face(profile_image_path)
        except Exception as e:
            print(f"Error aligning profile image: {str(e)}")
            aligned_profile = None
            
        try:
            aligned_live = align_face(live_image_path)
        except Exception as e:
            print(f"Error aligning live image: {str(e)}")
            aligned_live = None
        
        # If alignment failed, use original images
        profile_path = aligned_profile if aligned_profile else profile_image_path
        live_path = aligned_live if aligned_live else live_image_path
        
        # Step 2: Preprocess images
        processed_profile = preprocess_image(profile_path)
        processed_live = preprocess_image(live_path)
        
        # If preprocessing failed, use aligned or original images
        profile_path = processed_profile if processed_profile else profile_path
        live_path = processed_live if processed_live else live_path
        
        # Step 3: Calculate encodings directly if previous steps failed
        profile_encodings = calculate_multiple_encodings(profile_path)
        live_encodings = calculate_multiple_encodings(live_path)
        
        # Fallback to direct encoding if the previous method failed
        if not profile_encodings:
            profile_img = face_recognition.load_image_file(profile_image_path)
            profile_encoding = face_recognition.face_encodings(profile_img)
            if profile_encoding:
                profile_encodings = [profile_encoding[0]]
        
        if not live_encodings:
            live_img = face_recognition.load_image_file(live_image_path)
            live_encoding = face_recognition.face_encodings(live_img)
            if live_encoding:
                live_encodings = [live_encoding[0]]
        
        if not profile_encodings or not live_encodings:
            return {"match": False, "confidence": 0, "message": "Couldn't encode faces"}
        
        # Step 4: Calculate all pairwise distances
        distances = []
        for prof_enc in profile_encodings:
            for live_enc in live_encodings:
                distance = face_recognition.face_distance([prof_enc], live_enc)[0]
                distances.append(float(distance))  # Convert to Python float
        
        # Step 5: Use the minimum distance (best match)
        min_distance = min(distances)
        confidence = float(1 - min_distance)  # Convert to Python float
        
        # Step 6: Average of top 3 distances if we have enough pairs
        if len(distances) >= 3:
            top_distances = sorted(distances)[:3]
            avg_confidence = float(1 - sum(top_distances) / len(top_distances))
            # Blend the minimum and average for more stability
            confidence = 0.7 * confidence + 0.3 * avg_confidence
        
        # Step 7: Ensure all values are Python native types for JSON serialization
        match = bool(confidence >= threshold)  # Convert to Python bool
        
        detailed_results = {
            "min_distance": float(min_distance),
            "confidence": float(confidence),
            "num_profile_encodings": int(len(profile_encodings)),
            "num_live_encodings": int(len(live_encodings)),
            "match": match,
            "threshold": float(threshold)
        }
        
        return detailed_results
    except Exception as e:
        return {"match": False, "confidence": 0.0, "error": str(e)}

# Change the profile image path to be configurable via environment variable
PROFILE_IMAGE_PATH = os.environ.get('PROFILE_IMAGE_PATH', 'uploads/profile.jpg')

# Modify your verify_face function to use this path
@app.route('/verify', methods=['POST'])
def verify_face():
    """API to verify if two faces match"""
    if 'live_image' not in request.files:
        return jsonify({"error": "live_image is required!"}), 400

    live_image = request.files['live_image']
    # Use the configurable path instead of hardcoded one
    profile_image_path = PROFILE_IMAGE_PATH

    try:
        # Save the live image
        live_image_path = save_image(live_image, "live.jpg")

        # Compare faces with advanced technique
        result = verify_identity(profile_image_path, live_image_path)

        # Return detailed result
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Change the run command at the bottom of the file
if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Run the app on 0.0.0.0 to make it accessible outside the container
    app.run(host='0.0.0.0', port=port)