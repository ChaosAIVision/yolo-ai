import os
import logging

# CUDA config - MUST be set before importing torch
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES")
if CUDA_VISIBLE_DEVICES is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

logger = logging.getLogger(__name__)

# Model paths
MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "/home/chaos/Documents/chaos/production/yolo-ai/weights/best.pt")
ONNX_OUTPUT_DIR = os.getenv("ONNX_OUTPUT_DIR", "/home/chaos/Documents/chaos/production/yolo-ai/weights/")
ONNX_OPSET = int(os.getenv("ONNX_OPSET", "17"))

# Inference configs
CONF_THRES = float(os.getenv("CONF_THRES", "0.20"))
IOU_THRES = float(os.getenv("IOU_THRES", "0.3"))
MAX_MP = float(os.getenv("MAX_MP", "12.0"))
MAX_BATCH = int(os.getenv("MAX_BATCH", "8"))
USE_HALF = os.getenv("USE_HALF", "1") == "1"

# BentoML configs
BENTO_SERVICE_NAME = os.getenv("BENTO_SERVICE_NAME", "yolov8-service")
BENTO_MODEL_NAME = os.getenv("BENTO_MODEL_NAME", "yolov8-onnx")
BENTO_ENDPOINT_URL = os.getenv("BENTO_ENDPOINT_URL", "http://localhost:3000")

# Stream configs
FPS_LIMIT = int(os.getenv("FPS_LIMIT", "20"))
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "3"))  # Process every 3rd frame

# WebRTC configs
# Default STUN server helps discover server-reflexive candidates behind NAT
STUN_SERVER_URL = os.getenv("STUN_SERVER_URL", "stun:stun.l.google.com:19302")
TURN_SERVER_URL = os.getenv("TURN_SERVER_URL", "")
TURN_SERVER_USERNAME = os.getenv("TURN_SERVER_USERNAME", "")
TURN_SERVER_CREDENTIAL = os.getenv("TURN_SERVER_CREDENTIAL", "")

# CORS configs
# Include dev server ports 8081 and 8082 by default (localhost and 127.0.0.1)
CORS_ALLOWED_ORIGINS = os.getenv(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:8081,http://localhost:8082,http://localhost:8080,http://127.0.0.1:8081,http://127.0.0.1:8082,http://127.0.0.1:8080"
).split(",")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# CLASS_NAMES = {

#     0:"person",
#     1:"helmet",
#     2:"vest",
#     3:"shoes"
# }


CLASS_NAMES = {
    0:"Dust Mask",
    1:"Eye Wear",
    2:"Glove",
    3:"Jacket",
    4:"Protective Boots",
    5:"Protective Helmet",
    6:"Shield"
}