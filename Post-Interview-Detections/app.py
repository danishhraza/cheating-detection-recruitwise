from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
import uvicorn
import os
import tempfile
import boto3
import requests
import time
import json
import logging
import threading
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from botocore.config import Config
from botocore import UNSIGNED
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic

from object_detect import ExamCheatingDetector
from gaze_detect import detect_looking_away_violations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
KAFKA_INPUT_TOPIC = os.environ.get('KAFKA_INPUT_TOPIC', 'exam-videos')
KAFKA_OUTPUT_TOPIC = os.environ.get('KAFKA_OUTPUT_TOPIC', 'exam-results')
KAFKA_CONSUMER_GROUP = os.environ.get('KAFKA_CONSUMER_GROUP', 'exam-processor')

app = FastAPI(title="Exam Cheating Detection API")

# Store recent results in memory (for demo purposes)
# In production, consider using Redis or a database
results_cache = {}

def store_result(result):
    """Store a result in the cache"""
    request_id = result.get("request_id")
    if request_id:
        results_cache[request_id] = result
        # Limit cache size
        if len(results_cache) > 100:
            oldest_key = next(iter(results_cache))
            results_cache.pop(oldest_key)

# Kafka producer configuration
producer_config = {
    'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
    'client.id': 'exam-cheating-api-producer'
}

# Kafka consumer configuration
consumer_config = {
    'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
    'group.id': KAFKA_CONSUMER_GROUP,
    'auto.offset.reset': 'earliest'
}

class VideoRequest(BaseModel):
    video_url: HttpUrl
    request_id: Optional[str] = None
    use_cuda: bool = True
    half_precision: bool = False
    batch_size: int = 1
    gaze_threshold: float = 0.35
    alert_interval_seconds: float = 2.0

class ResultStatus(BaseModel):
    request_id: str
    status: str
    result: Optional[Dict] = None
    error: Optional[str] = None

# Store recent results in memory (for demo purposes)
# In production, consider using Redis or a database
results_cache = {}

class CheatingDetectionResponse(BaseModel):
    request_id: str
    object_violations: Dict[str, List[float]]
    gaze_violations: List[Dict[str, Any]]
    processing_time: float
    status: str = "completed"

@app.get("/results/{request_id}", response_model=ResultStatus)
async def get_result(request_id: str):
    """
    Get the status and result of a processing job
    """
    # Check if result is in cache
    if request_id in results_cache:
        result = results_cache[request_id]
        return ResultStatus(
            request_id=request_id,
            status=result.get("status", "unknown"),
            result=result if result.get("status") == "completed" else None,
            error=result.get("error")
        )
    
    # If not in cache, check Kafka (last 100 messages)
    consumer = Consumer({
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        'group.id': f'result-fetch-{int(time.time())}',
        'auto.offset.reset': 'latest'
    })
    
    try:
        consumer.subscribe([KAFKA_OUTPUT_TOPIC])
        
        # Poll for messages
        found = False
        for _ in range(100):  # Limit to 100 messages
            msg = consumer.poll(0.5)
            if msg is None:
                continue
                
            if msg.error():
                continue
                
            try:
                value = json.loads(msg.value())
                store_result(value)  # Store in cache for future
                
                if value.get("request_id") == request_id:
                    found = True
                    return ResultStatus(
                        request_id=request_id,
                        status=value.get("status", "unknown"),
                        result=value if value.get("status") == "completed" else None,
                        error=value.get("error")
                    )
            except:
                pass
            
        if not found:
            return ResultStatus(
                request_id=request_id,
                status="not_found"
            )
    finally:
        consumer.close()

def cleanup_file(file_path: str):
    """Background task to clean up temporary files"""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Deleted temporary file: {file_path}")
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {str(e)}")

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

def delivery_report(err, msg):
    """Kafka delivery report callback"""
    if err is not None:
        logger.error(f"Message delivery failed: {err}")
    else:
        logger.info(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")

def process_video_task(video_request: dict):
    """Process a video detection task from Kafka queue"""
    request_id = video_request.get('request_id', f"req-{time.time()}")
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_filename = temp_file.name
    temp_file.close()

    start_time = time.time()
    
    try:
        logger.info(f"[{request_id}] Downloading video from {video_request['video_url']}")
        download_file_from_s3_or_url(video_request['video_url'], temp_filename)

        logger.info(f"[{request_id}] Running object detection...")
        import torch  # Import inside function to avoid issues if CUDA isn't used elsewhere
        detector = ExamCheatingDetector(
            video_path=temp_filename,
            output_dir=tempfile.gettempdir(),
            use_cuda=video_request.get('use_cuda', True) and torch.cuda.is_available(),
            half_precision=video_request.get('half_precision', False),
            batch_size=video_request.get('batch_size', 1)
        )

        detector.process_video()
        detector.consolidate_violations()
        object_results = detector.violations

        logger.info(f"[{request_id}] Running gaze detection...")
        gaze_violations = detect_looking_away_violations(
            video_path=temp_filename,
            gaze_threshold=video_request.get('gaze_threshold', 0.35),
            alert_interval_seconds=video_request.get('alert_interval_seconds', 2.0)
        )

        processing_time = time.time() - start_time
        
        # Send results to Kafka output topic
        result = {
            "request_id": request_id,
            "object_violations": object_results,
            "gaze_violations": gaze_violations,
            "processing_time": processing_time,
            "status": "completed"
        }
        
        # Store in cache
        store_result(result)
        
        producer = Producer(producer_config)
        producer.produce(
            KAFKA_OUTPUT_TOPIC,
            json.dumps(result).encode('utf-8'),
            key=request_id,
            callback=delivery_report
        )
        producer.flush(timeout=5.0)
        
        logger.info(f"[{request_id}] Processing completed in {processing_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"[{request_id}] Error processing video: {str(e)}")
        # Send error message to Kafka
        error_result = {
            "request_id": request_id,
            "status": "error",
            "error": str(e)
        }
        
        # Store in cache
        store_result(error_result)
        
        producer = Producer(producer_config)
        producer.produce(
            KAFKA_OUTPUT_TOPIC,
            json.dumps(error_result).encode('utf-8'),
            key=request_id,
            callback=delivery_report
        )
        producer.flush(timeout=5.0)
    
    finally:
        cleanup_file(temp_filename)

@app.post("/submit-video", status_code=202)
async def submit_video(request: VideoRequest, background_tasks: BackgroundTasks):
    """
    Submit a video for asynchronous cheating detection via Kafka
    """
    try:
        if not request.request_id:
            request.request_id = f"req-{int(time.time() * 1000)}"
        
        # Convert request to dict for kafka
        # This converts HttpUrl to a string and handles serialization properly
        video_request = {
            "request_id": request.request_id,
            "video_url": str(request.video_url),  # Convert HttpUrl to string
            "use_cuda": request.use_cuda,
            "half_precision": request.half_precision,
            "batch_size": request.batch_size,
            "gaze_threshold": request.gaze_threshold,
            "alert_interval_seconds": request.alert_interval_seconds
        }
        
        # Try to use Kafka
        kafka_available = True
        try:
            # Send to Kafka queue
            producer = Producer(producer_config)
            producer.produce(
                KAFKA_INPUT_TOPIC,
                json.dumps(video_request).encode('utf-8'),
                key=request.request_id,
                callback=delivery_report
            )
            producer.flush(timeout=5.0)  # Add timeout for flush
            logger.info(f"Successfully submitted request {request.request_id} to Kafka")
        except Exception as ke:
            logger.warning(f"Kafka unavailable: {str(ke)}. Falling back to direct processing.")
            kafka_available = False
            
        if kafka_available:
            return {"request_id": request.request_id, "status": "submitted"}
        else:
            # FALLBACK: If Kafka is unavailable, process directly
            logger.info(f"Processing request {request.request_id} directly (Kafka fallback)")
            background_tasks.add_task(process_video_task, video_request)
            return {"request_id": request.request_id, "status": "processing_directly"}
            
    except Exception as e:
        logger.exception(f"Error in submit_video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {"api": "ok"}
    
    # Check Kafka connection
    try:
        admin_client = AdminClient({'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS})
        metadata = admin_client.list_topics(timeout=5)
        if metadata:
            status["kafka"] = "ok"
        else:
            status["kafka"] = "error"
    except Exception as e:
        status["kafka"] = f"error: {str(e)}"
        
    return status

def kafka_consumer_thread():
    """Background thread for Kafka message consumption"""
    logger.info("Starting Kafka consumer thread")
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            consumer = Consumer(consumer_config)
            consumer.subscribe([KAFKA_INPUT_TOPIC])
            
            logger.info(f"Successfully subscribed to topic {KAFKA_INPUT_TOPIC}")
            retry_count = 0  # Reset retry count on successful connection
            
            try:
                while True:
                    msg = consumer.poll(1.0)
                    if msg is None:
                        continue
                        
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            logger.info(f"Reached end of partition {msg.topic()} [{msg.partition()}]")
                        else:
                            logger.error(f"Kafka consumer error: {msg.error()}")
                            break
                    else:
                        try:
                            logger.info(f"Received message: {msg.value()[:100]}...")
                            video_request = json.loads(msg.value())
                            # Process in a separate thread to allow consumer to continue polling
                            threading.Thread(
                                target=process_video_task,
                                args=(video_request,)
                            ).start()
                        except Exception as e:
                            logger.error(f"Error processing message: {str(e)}")
                            
            except KeyboardInterrupt:
                logger.info("Shutting down Kafka consumer")
            finally:
                consumer.close()
                
        except KafkaException as e:
            retry_count += 1
            logger.error(f"Error connecting to Kafka (attempt {retry_count}/{max_retries}): {str(e)}")
            
            if "Unknown topic" in str(e):
                # Topic doesn't exist yet, try to create it
                logger.info("Attempting to create missing topics")
                create_topics_if_not_exist()
            
            # Wait before retrying
            wait_time = 5 * retry_count  # Exponential backoff
            logger.info(f"Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)
            
    logger.error("Kafka consumer thread stopped after maximum retries")

def create_topics_if_not_exist():
    """Create Kafka topics if they don't exist"""
    max_retries = 5
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to create Kafka topics (attempt {attempt+1}/{max_retries})")
            
            admin_client = AdminClient({'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS})
            
            # Explicitly create topics without checking if they exist first
            # This is more reliable in a containerized environment
            topics_to_create = [
                NewTopic(
                    KAFKA_INPUT_TOPIC,
                    num_partitions=1,
                    replication_factor=1
                ),
                NewTopic(
                    KAFKA_OUTPUT_TOPIC,
                    num_partitions=1,
                    replication_factor=1
                )
            ]
            
            # Create topics
            futures = admin_client.create_topics(topics_to_create)
            
            # Wait for operation to complete
            for topic, future in futures.items():
                try:
                    future.result(timeout=10)  # Wait with timeout
                    logger.info(f"Topic {topic} created or already exists")
                except Exception as e:
                    # If topic already exists, that's fine
                    if "already exists" in str(e) or "TopicExistsError" in str(e):
                        logger.info(f"Topic {topic} already exists")
                    else:
                        logger.warning(f"Error creating topic {topic}: {e}")
            
            # Verify topics exist by listing them
            metadata = admin_client.list_topics(timeout=10)
            topics = metadata.topics
            
            logger.info(f"Available Kafka topics: {', '.join(topics.keys())}")
            
            if KAFKA_INPUT_TOPIC in topics and KAFKA_OUTPUT_TOPIC in topics:
                logger.info("All required Kafka topics are available")
                return True
            else:
                missing = []
                if KAFKA_INPUT_TOPIC not in topics:
                    missing.append(KAFKA_INPUT_TOPIC)
                if KAFKA_OUTPUT_TOPIC not in topics:
                    missing.append(KAFKA_OUTPUT_TOPIC)
                logger.warning(f"Some topics are still missing: {', '.join(missing)}")
                
        except Exception as e:
            logger.error(f"Error creating Kafka topics: {e}")
        
        if attempt < max_retries - 1:
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    logger.error("Failed to create Kafka topics after multiple attempts")
    return False

@app.on_event("startup")
async def startup_event():
    """Start Kafka consumer on application startup"""
    # First, try to create Kafka topics
    topics_created = create_topics_if_not_exist()
    
    if topics_created:
        logger.info("Kafka topics created successfully.")
    else:
        logger.warning("Could not create Kafka topics. Some features may not work correctly.")
    
    # Start Kafka consumer in a background thread
    threading.Thread(target=kafka_consumer_thread, daemon=True).start()
    
    logger.info("Application startup complete")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)