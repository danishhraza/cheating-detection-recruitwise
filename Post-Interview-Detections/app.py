from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
import uvicorn
import os
import tempfile
import boto3
import requests
import time
from typing import List, Dict, Any
from urllib.parse import urlparse
from botocore.config import Config
from botocore import UNSIGNED

from object_detect import ExamCheatingDetector
from gaze_detect import detect_looking_away_violations

app = FastAPI(title="Exam Cheating Detection API")

class VideoRequest(BaseModel):
    video_url: HttpUrl
    use_cuda: bool = True
    half_precision: bool = False
    batch_size: int = 1
    gaze_threshold: float = 0.35
    alert_interval_seconds: float = 2.0

class CheatingDetectionResponse(BaseModel):
    object_violations: Dict[str, List[float]]
    gaze_violations: List[Dict[str, Any]]
    processing_time: float

def cleanup_file(file_path: str):
    """Background task to clean up temporary files"""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Deleted temporary file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {str(e)}")

def download_file_from_s3_or_url(url: str, local_path: str):
    """Download a file from public S3 (s3://) or HTTP(S) URL"""
    if url.startswith("s3://"):
        parsed = urlparse(url)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        s3.download_file(bucket, key, local_path)

    elif url.startswith("http://") or url.startswith("https://"):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        raise ValueError(f"Unsupported URL format: {url}")

@app.post("/detect-cheating", response_model=CheatingDetectionResponse)
async def detect_cheating(request: VideoRequest, background_tasks: BackgroundTasks):
    """
    Detect cheating behaviors in an exam video using object detection and gaze analysis.
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_filename = temp_file.name
    temp_file.close()

    start_time = time.time()

    try:
        print(f"Downloading video from {request.video_url}")
        download_file_from_s3_or_url(str(request.video_url), temp_filename)

        print("Running object detection...")
        import torch  # Import inside function to avoid issues if CUDA isn't used elsewhere
        detector = ExamCheatingDetector(
            video_path=temp_filename,
            output_dir=tempfile.gettempdir(),
            use_cuda=request.use_cuda and torch.cuda.is_available(),
            half_precision=request.half_precision,
            batch_size=request.batch_size
        )

        detector.process_video()
        detector.consolidate_violations()
        object_results = detector.violations

        print("Running gaze detection...")
        gaze_violations = detect_looking_away_violations(
            video_path=temp_filename,
            gaze_threshold=request.gaze_threshold,
            alert_interval_seconds=request.alert_interval_seconds
        )

        processing_time = time.time() - start_time
        background_tasks.add_task(cleanup_file, temp_filename)

        return CheatingDetectionResponse(
            object_violations=object_results,
            gaze_violations=gaze_violations,
            processing_time=processing_time
        )

    except Exception as e:
        background_tasks.add_task(cleanup_file, temp_filename)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
