#!/usr/bin/env python
"""
Standalone Kafka worker for processing exam cheating detection requests.
This worker runs independently and can be scaled horizontally.
"""

import os
import tempfile
import boto3
import requests
import time
import json
import logging
import threading
import signal
import sys
from urllib.parse import urlparse
from botocore.config import Config
from botocore import UNSIGNED
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException

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
KAFKA_CONSUMER_GROUP = os.environ.get('KAFKA_CONSUMER_GROUP', 'exam-processor-worker')

# Kafka producer configuration
producer_config = {
    'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
    'client.id': 'exam-cheating-worker-producer'
}

# Kafka consumer configuration
consumer_config = {
    'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
    'group.id': KAFKA_CONSUMER_GROUP,
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': False,  # For manual commit after processing
}

def cleanup_file(file_path: str):
    """Clean up temporary files"""
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

def process_video_request(video_request):
    """Process a video detection task"""
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
        
        producer = Producer(producer_config)
        producer.produce(
            KAFKA_OUTPUT_TOPIC,
            json.dumps(result).encode('utf-8'),
            key=request_id,
            callback=delivery_report
        )
        producer.flush()
        
        logger.info(f"[{request_id}] Processing completed in {processing_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"[{request_id}] Error processing video: {str(e)}")
        # Send error message to Kafka
        error_result = {
            "request_id": request_id,
            "status": "error",
            "error": str(e)
        }
        producer = Producer(producer_config)
        producer.produce(
            KAFKA_OUTPUT_TOPIC,
            json.dumps(error_result).encode('utf-8'),
            key=request_id,
            callback=delivery_report
        )
        producer.flush()
    
    finally:
        cleanup_file(temp_filename)

def main():
    """Main worker function"""
    logger.info(f"Starting Kafka worker. Consumer group: {KAFKA_CONSUMER_GROUP}")
    logger.info(f"Listening for messages on topic: {KAFKA_INPUT_TOPIC}")
    
    # Set up signal handling for graceful shutdown
    running = True
    
    def signal_handler(sig, frame):
        nonlocal running
        logger.info("Received shutdown signal, finishing up...")
        running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create consumer and subscribe to topic
    consumer = Consumer(consumer_config)
    consumer.subscribe([KAFKA_INPUT_TOPIC])
    
    try:
        while running:
            msg = consumer.poll(1.0)
            
            if msg is None:
                continue
                
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.info(f"Reached end of partition {msg.topic()} [{msg.partition()}]")
                else:
                    logger.error(f"Error: {msg.error()}")
            else:
                try:
                    logger.info(f"Received message: {msg.value()[:100]}...")
                    video_request = json.loads(msg.value())
                    process_video_request(video_request)
                    # Manually commit offset after successful processing
                    consumer.commit(msg)
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        logger.info("Closing consumer")
        consumer.close()
        
if __name__ == "__main__":
    main()