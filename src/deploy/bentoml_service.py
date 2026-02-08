"""BentoML service for YOLO model deployment using Triton Inference Server or Ultralytics."""
import os
import torch
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# Set CUDA_VISIBLE_DEVICES BEFORE importing bentoml
# This ensures BentoML only sees the specified GPU(s)
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES")
if CUDA_VISIBLE_DEVICES is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

import logging
from typing import Any, Dict, List

import bentoml
from bentoml.legacy import Runnable, Runner, Service as LegacyService
try:
    from bentoml.io import Image as BentoImage
    from bentoml.io import JSON as BentoJSON
except ImportError:
    from bentoml import Image as BentoImage
    from bentoml import JSON as BentoJSON
import numpy as np
from PIL import Image

from src.config import (
    MODEL_WEIGHTS_PATH,
    CONF_THRES,
    IOU_THRES,
    MAX_MP,
    MAX_BATCH,
    BENTO_SERVICE_NAME,
    BENTO_MODEL_NAME,
    CLASS_NAMES,
    TRITON_URL,
    logger,
)
from src.triton.auto_config import TRITON_MODEL_NAME

logger = logging.getLogger(__name__)


class YOLOONNXRunnable(Runnable):
    """BentoML Runnable for YOLO inference using Triton Inference Server or Ultralytics."""
    
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True
    
    def __init__(self, model_path: str = None):
        
        try:
            from torch.serialization import add_safe_globals
            import ultralytics.nn.tasks as ul_tasks
            add_safe_globals([ul_tasks.DetectionModel])
        except Exception:
            pass
        
        # Determine model source: Triton URL or local file
        if TRITON_URL:
            # Use Triton Inference Server
            # TRITON_URL format: http://triton-yolo:8000
            # Model URL: TRITON_URL + model_name = http://triton-yolo:8000/yolov8n
            triton_base_url = TRITON_URL.rstrip('/')
            if not triton_base_url.startswith(('http://', 'https://')):
                triton_base_url = f"http://{triton_base_url}"
            
            model_source = f"{triton_base_url}/{TRITON_MODEL_NAME}"
            logger.info(f"Loading YOLO model from Triton: {model_source}")
        else:
            # Use local model files
            model_path = model_path or MODEL_WEIGHTS_PATH
            
            # If model_path is a directory, find model file
            if os.path.isdir(model_path):
                pt_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
                onnx_files = [f for f in os.listdir(model_path) if f.endswith('.onnx')]
                
                # Prefer ONNX if available (smaller, faster)
                if onnx_files:
                    model_path = os.path.join(model_path, onnx_files[0])
                    logger.info(f"Found ONNX model in directory: {model_path}")
                elif pt_files:
                    model_path = os.path.join(model_path, pt_files[0])
                    logger.info(f"Found PyTorch model in directory: {model_path}")
                else:
                    raise FileNotFoundError(f"No model files (.pt or .onnx) found in directory: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model weights not found: {model_path}")
            
            model_source = model_path
            logger.info(f"Loading YOLO model from local file: {model_source}")
        
        self.is_onnx = model_source.endswith('.onnx') or TRITON_URL is not None
        
        # Clear CUDA cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible:
                logger.info(f"CUDA_VISIBLE_DEVICES={cuda_visible}")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                logger.info(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # Use cuda:0 (when CUDA_VISIBLE_DEVICES is set, specified GPU becomes cuda:0)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device_idx = 0 if device.startswith("cuda") else None
        self.device = device
        
        # Load model - YOLO supports both file paths and Triton HTTP URLs
        self.model = YOLO(model_source, task="detect")
        
        # Log memory usage after loading
        if torch.cuda.is_available() and self.device_idx is not None:
            allocated = torch.cuda.memory_allocated(self.device_idx) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device_idx) / 1024**3
            logger.info(f"GPU {self.device_idx} memory after model load - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        
        self.class_names = CLASS_NAMES
    
    @Runnable.method(batchable=True, batch_dim=(0, 0))
    def infer(self, imgs: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Batch inference on images.
        
        Args:
            imgs: List of HWC uint8 numpy arrays
            
        Returns:
            List of detection results
        """
        # Clear cache before inference to avoid OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self.device_idx is not None:
                torch.cuda.synchronize(self.device_idx)
        
        # YOLO handles both local files and Triton URLs automatically
        if self.is_onnx:
            results = self.model.predict(
                source=imgs, 
                conf=CONF_THRES, 
                iou=IOU_THRES, 
                device=self.device_idx if self.device_idx is not None else "cpu",
                verbose=False
            )
        else:
            results = self.model.predict(
                source=imgs, conf=CONF_THRES, iou=IOU_THRES, verbose=False
            )
        return [self._results_to_json(r) for r in results]
    
    def _results_to_json(self, result) -> Dict[str, Any]:
        """Convert Ultralytics Results to simple JSON-friendly structure."""
        detections: List[Dict[str, Any]] = []
        
        if getattr(result, "boxes", None) is not None and result.boxes is not None:
            boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(float)
            scores = result.boxes.conf.detach().cpu().numpy().astype(float)
            class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
            
            for i in range(boxes_xyxy.shape[0]):
                class_id = int(class_ids[i])
                detections.append({
                    "xyxy": boxes_xyxy[i].tolist(),
                    "confidence": float(scores[i]),
                    "class_id": class_id,
                    "class_name": self.class_names.get(class_id, str(class_id)),
                })
        
        h, w = None, None
        if getattr(result, "orig_shape", None) is not None:
            h, w = int(result.orig_shape[0]), int(result.orig_shape[1])
        
        return {"detections": detections, "image_shape": [h, w]}


def create_bentoml_service(model_path: str = None):
    """Create and return BentoML service."""
    runner = Runner(
        YOLOONNXRunnable,
        name=BENTO_MODEL_NAME,
        runnable_init_params={"model_path": model_path},
        max_batch_size=MAX_BATCH,
        max_latency_ms=50,
    )
    
    svc = LegacyService(
        name=BENTO_SERVICE_NAME,
        runners=[runner],
    )
    
    @svc.api(input=BentoImage(), output=BentoJSON())
    def predict(image: Image.Image) -> Dict[str, Any]:
        """Predict on single image."""
        img, (w, h) = _validate_image(image)
        img_array = np.asarray(img)
        results = runner.infer.run([img_array])
        return results[0]
    
    @svc.api(input=BentoJSON(), output=BentoJSON())
    def health(_: Dict[str, Any]) -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}
    
    @svc.api(input=BentoJSON(), output=BentoJSON())
    def ready(_: Dict[str, Any]) -> Dict[str, str]:
        """Readiness check endpoint."""
        if runner is None:
            return {"status": "not_ready", "message": "Runner not initialized"}
        return {"status": "ready"}
    
    return svc


def _validate_image(img: Image.Image) -> tuple:
    """Validate image size and convert to RGB."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    w, h = img.size
    mp = (w * h) / 1_000_000.0
    
    if mp > MAX_MP:
        raise ValueError(f"Image too large: {w}x{h} ~ {mp:.2f}MP > MAX_MP={MAX_MP}MP")
    
    return img, (w, h)


if __name__ == "__main__":
    # Use TRITON_URL if set, otherwise use MODEL_WEIGHTS_PATH
    model_path = None  # Will use MODEL_WEIGHTS_PATH from config if TRITON_URL is not set
    
    svc = create_bentoml_service(model_path)
    logger.info(f"BentoML service created: {BENTO_SERVICE_NAME}")
