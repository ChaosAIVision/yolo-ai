"""Auto quantization module for YOLO models.
Supports multiple quantization formats: ONNX, TensorRT, OpenVINO.
"""
import os
import sys
import logging
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from ultralytics import YOLO
import torch

from src.config import MODEL_WEIGHTS_PATH, ONNX_OUTPUT_DIR, ONNX_OPSET, logger

load_dotenv()

# Get quantization config
QUANTIZATION_TYPE = os.getenv("QUANTIZATION_TYPE", "onnx").lower()

# Ensure onnx module is imported correctly (avoid conflict with filename)
if 'onnx' in sys.modules and not hasattr(sys.modules['onnx'], '__version__'):
    del sys.modules['onnx']


def convert_to_onnx(
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    opset: Optional[int] = None,
    simplify: bool = True,
    dynamic: bool = True,
    half: bool = False,
) -> str:
    """
    Convert YOLO model to ONNX format.
    
    Args:
        model_path: Path to YOLO model weights (.pt file)
        output_dir: Directory to save ONNX model
        opset: ONNX opset version
        simplify: Whether to simplify ONNX model
        dynamic: Whether to use dynamic input shapes
        half: Whether to use FP16 precision
        
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
    
    logger.info(f"Converting to ONNX (opset={opset}, simplify={simplify}, dynamic={dynamic}, half={half})")
    
    model.export(
        format="onnx",
        imgsz=640,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
        half=half,
    )
    
    exported_path = Path(model_path).parent / f"{model_name}.onnx"
    if not exported_path.exists():
        raise RuntimeError(f"ONNX export failed: {exported_path} not found")
    
    if exported_path != onnx_path:
        shutil.move(str(exported_path), str(onnx_path))
        logger.info(f"Moved ONNX model to: {onnx_path}")
    
    logger.info(f"ONNX model saved to: {onnx_path}")
    return str(onnx_path)


def convert_to_tensorrt(
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    half: bool = True,
    workspace: int = 4,
    simplify: bool = True,
) -> str:
    """
    Convert YOLO model to TensorRT format.
    
    Args:
        model_path: Path to YOLO model weights (.pt file)
        output_dir: Directory to save TensorRT model
        half: Whether to use FP16 precision (recommended for TensorRT)
        workspace: TensorRT workspace size in GB
        simplify: Whether to simplify model before conversion
        
    Returns:
        Path to converted TensorRT model
        
    Raises:
        FileNotFoundError: If model_path doesn't exist
        RuntimeError: If conversion fails
    """
    model_path = model_path or MODEL_WEIGHTS_PATH
    output_dir = output_dir or ONNX_OUTPUT_DIR
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT conversion requires CUDA support")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_name = Path(model_path).stem
    engine_path = output_path / f"{model_name}.engine"
    
    logger.info(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    logger.info(f"Converting to TensorRT (half={half}, workspace={workspace}GB, simplify={simplify})")
    
    model.export(
        format="engine",
        imgsz=640,
        half=half,
        workspace=workspace,
        simplify=simplify,
    )
    
    exported_path = Path(model_path).parent / f"{model_name}.engine"
    if not exported_path.exists():
        raise RuntimeError(f"TensorRT export failed: {exported_path} not found")
    
    if exported_path != engine_path:
        shutil.move(str(exported_path), str(engine_path))
        logger.info(f"Moved TensorRT model to: {engine_path}")
    
    logger.info(f"TensorRT model saved to: {engine_path}")
    return str(engine_path)


def convert_to_openvino(
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    half: bool = False,
    simplify: bool = True,
) -> str:
    """
    Convert YOLO model to OpenVINO format.
    
    Args:
        model_path: Path to YOLO model weights (.pt file)
        output_dir: Directory to save OpenVINO model
        half: Whether to use FP16 precision
        simplify: Whether to simplify model before conversion
        
    Returns:
        Path to converted OpenVINO model directory
        
    Raises:
        FileNotFoundError: If model_path doesn't exist
        RuntimeError: If conversion fails
    """
    model_path = model_path or MODEL_WEIGHTS_PATH
    output_dir = output_dir or ONNX_OUTPUT_DIR
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_name = Path(model_path).stem
    openvino_dir = output_path / f"{model_name}_openvino"
    
    logger.info(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    logger.info(f"Converting to OpenVINO (half={half}, simplify={simplify})")
    
    model.export(
        format="openvino",
        imgsz=640,
        half=half,
        simplify=simplify,
    )
    
    exported_path = Path(model_path).parent / f"{model_name}_openvino"
    if not exported_path.exists():
        raise RuntimeError(f"OpenVINO export failed: {exported_path} not found")
    
    if exported_path != openvino_dir:
        if openvino_dir.exists():
            shutil.rmtree(openvino_dir)
        shutil.move(str(exported_path), str(openvino_dir))
        logger.info(f"Moved OpenVINO model to: {openvino_dir}")
    
    logger.info(f"OpenVINO model saved to: {openvino_dir}")
    return str(openvino_dir)


def auto_quantize(
    quantization_type: Optional[str] = None,
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    **kwargs
) -> str:
    """
    Automatically quantize YOLO model based on QUANTIZATION_TYPE.
    
    Args:
        quantization_type: Type of quantization ('onnx', 'tensorrt', 'openvino')
        model_path: Path to YOLO model weights (.pt file)
        output_dir: Directory to save quantized model
        **kwargs: Additional arguments for specific quantization types
        
    Returns:
        Path to quantized model
        
    Raises:
        ValueError: If quantization_type is not supported
        FileNotFoundError: If model_path doesn't exist
        RuntimeError: If quantization fails
    """
    quantization_type = (quantization_type or QUANTIZATION_TYPE).lower()
    
    logger.info(f"Starting auto quantization with type: {quantization_type}")
    
    if quantization_type == "onnx":
        return convert_to_onnx(
            model_path=model_path,
            output_dir=output_dir,
            opset=kwargs.get("opset"),
            simplify=kwargs.get("simplify", True),
            dynamic=kwargs.get("dynamic", True),
            half=kwargs.get("half", False),
        )
    elif quantization_type == "tensorrt":
        return convert_to_tensorrt(
            model_path=model_path,
            output_dir=output_dir,
            half=kwargs.get("half", True),
            workspace=kwargs.get("workspace", 4),
            simplify=kwargs.get("simplify", True),
        )
    elif quantization_type == "openvino":
        return convert_to_openvino(
            model_path=model_path,
            output_dir=output_dir,
            half=kwargs.get("half", False),
            simplify=kwargs.get("simplify", True),
        )
    else:
        raise ValueError(
            f"Unsupported quantization type: {quantization_type}. "
            f"Supported types: 'onnx', 'tensorrt', 'openvino'"
        )


if __name__ == "__main__":
    try:
        quantized_model_path = auto_quantize()
        logger.info(f"Quantization completed successfully: {quantized_model_path}")
    except Exception as e:
        logger.error(f"Quantization failed: {e}", exc_info=True)
        sys.exit(1)
