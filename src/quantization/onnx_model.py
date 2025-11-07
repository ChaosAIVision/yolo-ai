"""Convert YOLO model to ONNX format for production deployment."""
import os
import sys
import logging
from pathlib import Path

from ultralytics import YOLO
import torch

from src.config import MODEL_WEIGHTS_PATH, ONNX_OUTPUT_DIR, ONNX_OPSET, logger

# Ensure onnx module is imported correctly (avoid conflict with filename)
if 'onnx' in sys.modules and not hasattr(sys.modules['onnx'], '__version__'):
    del sys.modules['onnx']
import onnx

logger = logging.getLogger(__name__)


def convert_to_onnx(
    model_path: str = None,
    output_dir: str = None,
    opset: int = None,
    simplify: bool = True,
    dynamic: bool = False,
) -> str:
    """
    Convert YOLO model to ONNX format.
    
    Args:
        model_path: Path to YOLO model weights (.pt file)
        output_dir: Directory to save ONNX model
        opset: ONNX opset version
        simplify: Whether to simplify ONNX model
        dynamic: Whether to use dynamic input shapes
        
    Returns:
        Path to converted ONNX model
        
    Raises:
        FileNotFoundError: If model_path doesn't exist
        RuntimeError: If conversion fails
    """
    model_path = model_path or MODEL_WEIGHTS_PATH
    output_dir = output_dir or ONNX_OUTPUT_DIR
    opset = opset or ONNX_OPSET
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_name = Path(model_path).stem
    onnx_path = output_path / f"{model_name}.onnx"
    
    logger.info(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    logger.info(f"Converting to ONNX (opset={opset}, simplify={simplify}, dynamic={dynamic})")
    
    model.export(
        format="onnx",
        imgsz=640,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
        half=False,
    )
    
    exported_path = Path(model_path).parent / f"{model_name}.onnx"
    if not exported_path.exists():
        raise RuntimeError(f"ONNX export failed: {exported_path} not found")
    
    if exported_path != onnx_path:
        import shutil
        shutil.move(str(exported_path), str(onnx_path))
        logger.info(f"Moved ONNX model to: {onnx_path}")
    
    logger.info(f"ONNX model saved to: {onnx_path}")
    return str(onnx_path)


if __name__ == "__main__":
    onnx_model_path = convert_to_onnx()
    logger.info(f"Conversion completed: {onnx_model_path}")

