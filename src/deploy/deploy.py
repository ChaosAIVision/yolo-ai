"""Deploy YOLO ONNX model to BentoML."""
import os
import sys
import logging
from pathlib import Path

import bentoml

from src.config import (
    MODEL_WEIGHTS_PATH,
    ONNX_OUTPUT_DIR,
    BENTO_SERVICE_NAME,
    logger,
)
from src.quantization.onnx import convert_to_onnx

logger = logging.getLogger(__name__)


def deploy_to_bentoml(
    model_path: str = None,
    onnx_path: str = None,
    convert_first: bool = True,
) -> str:
    """
    Deploy YOLO model to BentoML.
    
    Args:
        model_path: Path to YOLO model weights (.pt file)
        onnx_path: Path to ONNX model (if already converted)
        convert_first: Whether to convert to ONNX first if onnx_path not provided
        
    Returns:
        BentoML service tag
        
    Raises:
        FileNotFoundError: If model files don't exist
        RuntimeError: If deployment fails
    """
    if onnx_path is None:
        if convert_first:
            logger.info("Converting model to ONNX first...")
            onnx_path = convert_to_onnx(model_path=model_path)
        else:
            raise ValueError("onnx_path must be provided if convert_first=False")
    
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    logger.info(f"Deploying ONNX model to BentoML: {onnx_path}")
    
    from src.deploy.bentoml_service import create_bentoml_service
    
    svc = create_bentoml_service(onnx_path)
    
    logger.info(f"BentoML service created: {BENTO_SERVICE_NAME}")
    logger.info("To build Bento, run: bentoml build")
    logger.info("To serve locally, run: bentoml serve")
    
    return BENTO_SERVICE_NAME


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    onnx_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if model_path is None:
        model_path = MODEL_WEIGHTS_PATH
    
    service_name = deploy_to_bentoml(
        model_path=model_path,
        onnx_path=onnx_path,
        convert_first=(onnx_path is None),
    )
    
    logger.info(f"Deployment completed: {service_name}")

