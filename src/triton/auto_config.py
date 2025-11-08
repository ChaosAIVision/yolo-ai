"""Auto configure Triton Inference Server model repository.
Automatically builds Triton config and moves quantized models to repository.
"""
import os
import sys
import logging
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.config import MODEL_WEIGHTS_PATH, ONNX_OUTPUT_DIR, MAX_BATCH, logger

load_dotenv()

# Get quantization config
QUANTIZATION_TYPE = os.getenv("QUANTIZATION_TYPE", "onnx").lower()

# Triton config paths
# Get the project root (assuming this file is in src/triton/)
_project_root = Path(__file__).parent.parent.parent
_default_repository = _project_root / "src" / "triton_config" / "model_repository"

TRITON_MODEL_REPOSITORY = os.getenv(
    "TRITON_MODEL_REPOSITORY",
    str(_default_repository)
)
TRITON_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "yolov8n")
TRITON_MODEL_VERSION = os.getenv("TRITON_MODEL_VERSION", "1")

# Map quantization type to Triton platform
PLATFORM_MAP = {
    "onnx": "onnxruntime_onnx",
    "tensorrt": "tensorrt_plan",
    "openvino": "openvino",
}

# Map quantization type to expected file extensions/patterns
MODEL_PATTERNS = {
    "onnx": ["*.onnx"],
    "tensorrt": ["*.engine"],
    "openvino": ["*.xml", "*.bin"],
}


def get_triton_platform(quantization_type: Optional[str] = None) -> str:
    """
    Get Triton platform based on quantization type.
    
    Args:
        quantization_type: Type of quantization ('onnx', 'tensorrt', 'openvino')
        
    Returns:
        Triton platform string
        
    Raises:
        ValueError: If quantization_type is not supported
    """
    quantization_type = (quantization_type or QUANTIZATION_TYPE).lower()
    
    if quantization_type not in PLATFORM_MAP:
        raise ValueError(
            f"Unsupported quantization type: {quantization_type}. "
            f"Supported types: {list(PLATFORM_MAP.keys())}"
        )
    
    return PLATFORM_MAP[quantization_type]


def find_quantized_model(
    quantization_type: Optional[str] = None,
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """
    Find the quantized model file/directory.
    
    Args:
        quantization_type: Type of quantization
        model_path: Original model path (to derive name)
        output_dir: Directory where quantized model was saved
        
    Returns:
        Path to quantized model
        
    Raises:
        FileNotFoundError: If model is not found
    """
    quantization_type = (quantization_type or QUANTIZATION_TYPE).lower()
    output_dir = output_dir or ONNX_OUTPUT_DIR
    model_path = model_path or MODEL_WEIGHTS_PATH
    
    model_name = Path(model_path).stem
    output_path = Path(output_dir)
    
    logger.info(f"Looking for {quantization_type} model: {model_name} in {output_dir}")
    
    if quantization_type == "onnx":
        # Look for .onnx file
        model_file = output_path / f"{model_name}.onnx"
        if not model_file.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {model_file}. "
                f"Please run quantization first."
            )
        return model_file
    
    elif quantization_type == "tensorrt":
        # Look for .engine file
        model_file = output_path / f"{model_name}.engine"
        if not model_file.exists():
            raise FileNotFoundError(
                f"TensorRT model not found: {model_file}. "
                f"Please run quantization first."
            )
        return model_file
    
    elif quantization_type == "openvino":
        # Look for OpenVINO directory
        model_dir = output_path / f"{model_name}_openvino"
        if not model_dir.exists() or not model_dir.is_dir():
            raise FileNotFoundError(
                f"OpenVINO model directory not found: {model_dir}. "
                f"Please run quantization first."
            )
        
        # Verify OpenVINO files exist
        xml_file = model_dir / f"{model_name}.xml"
        bin_file = model_dir / f"{model_name}.bin"
        
        if not xml_file.exists() or not bin_file.exists():
            # Try to find any .xml and .bin files
            xml_files = list(model_dir.glob("*.xml"))
            bin_files = list(model_dir.glob("*.bin"))
            
            if not xml_files or not bin_files:
                raise FileNotFoundError(
                    f"OpenVINO model files (.xml/.bin) not found in: {model_dir}"
                )
        
        return model_dir
    
    else:
        raise ValueError(f"Unsupported quantization type: {quantization_type}")


def create_triton_config(
    model_name: str,
    platform: str,
    quantization_type: str,
    max_batch_size: int = 8,
    input_dims: list = [3, 640, 640],  # Note: no batch dimension in Triton config
    input_dtype: str = "TYPE_FP32",
    gpu_count: int = 1,
    gpu_ids: list = [0],
    enable_dynamic_batching: bool = True,
    preferred_batch_sizes: list = [1, 2, 4, 8],
    max_queue_delay_microseconds: int = 100,
    enable_cuda_graphs: bool = True,
    enable_busy_wait_events: bool = True,
) -> str:
    """
    Create Triton config.pbtxt content.
    
    Args:
        model_name: Name of the model
        platform: Triton platform (e.g., 'onnxruntime_onnx', 'tensorrt_plan', 'openvino')
        quantization_type: Type of quantization
        max_batch_size: Maximum batch size (default: 8)
        input_dims: Input dimensions [channels, height, width] (no batch dimension)
        input_dtype: Input data type
        gpu_count: Number of GPU instances
        gpu_ids: List of GPU IDs
        enable_dynamic_batching: Enable dynamic batching (default: True)
        preferred_batch_sizes: Preferred batch sizes for dynamic batching
        max_queue_delay_microseconds: Max queue delay for dynamic batching
        enable_cuda_graphs: Enable CUDA graphs optimization
        enable_busy_wait_events: Enable busy wait events
        
    Returns:
        Config.pbtxt content as string
    """
    # Ensure input_dims doesn't include batch dimension
    # If user provides [batch, channels, height, width], remove batch
    if len(input_dims) == 4:
        input_dims = input_dims[1:]
    
    # Ensure max_batch_size > 0 when dynamic batching is enabled
    if enable_dynamic_batching and max_batch_size <= 0:
        max_batch_size = 8
        logger.warning(f"max_batch_size must be > 0 for dynamic batching. Setting to {max_batch_size}")
    
    config_lines = [
        f'name: "{model_name}"',
        f'platform: "{platform}"',
        f"max_batch_size: {max_batch_size}",
        "",
        "input [",
        "  {",
        f'    name: "images"',
        f"    data_type: {input_dtype}",
        f"    dims: [ {', '.join(map(str, input_dims))} ]",
        "  }",
        "]",
        "",
    ]
    
    # Add dynamic batching if enabled (always enabled by default)
    if enable_dynamic_batching:
        config_lines.extend([
            "dynamic_batching {",
            f"  preferred_batch_size: [ {', '.join(map(str, preferred_batch_sizes))} ]",
            f"  max_queue_delay_microseconds: {max_queue_delay_microseconds}",
            "}",
            "",
        ])
    
    # Add instance group for GPU
    config_lines.extend([
        "instance_group [",
        "  {",
        f"    count: {gpu_count}",
        "    kind: KIND_GPU",
        f"    gpus: [ {', '.join(map(str, gpu_ids))} ]",
        "  }",
        "]",
        "",
    ])
    
    # Add optimization settings (only for GPU platforms)
    if platform in ["onnxruntime_onnx", "tensorrt_plan"] and gpu_count > 0:
        config_lines.extend([
            "optimization {",
            "  cuda {",
            f"    graphs: {'true' if enable_cuda_graphs else 'false'}",
            f"    busy_wait_events: {'true' if enable_busy_wait_events else 'false'}",
            "  }",
            "}",
        ])
    
    return "\n".join(config_lines)


def setup_triton_model_repository(
    quantized_model_path: Path,
    quantization_type: Optional[str] = None,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    repository_dir: Optional[str] = None,
) -> str:
    """
    Setup Triton model repository by moving quantized model and creating config.
    
    Args:
        quantized_model_path: Path to quantized model file/directory
        quantization_type: Type of quantization
        model_name: Model name in Triton repository
        model_version: Model version (default: "1")
        repository_dir: Triton model repository directory
        
    Returns:
        Path to model directory in repository
        
    Raises:
        FileNotFoundError: If quantized_model_path doesn't exist
        RuntimeError: If setup fails
    """
    quantization_type = (quantization_type or QUANTIZATION_TYPE).lower()
    model_name = model_name or TRITON_MODEL_NAME
    model_version = model_version or TRITON_MODEL_VERSION
    repository_dir = repository_dir or TRITON_MODEL_REPOSITORY
    
    # Get absolute path
    repository_path = Path(repository_dir).resolve()
    model_dir = repository_path / model_name
    version_dir = model_dir / model_version
    
    logger.info(f"Setting up Triton model repository: {model_dir}")
    
    # Create directories
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # Get platform
    platform = get_triton_platform(quantization_type)
    
    # Move/copy model to repository
    if quantization_type == "openvino":
        # OpenVINO: Triton expects model.xml and model.bin in version directory
        logger.info(f"Copying OpenVINO model files to: {version_dir}")
        
        # Find .xml and .bin files in OpenVINO directory
        xml_files = list(quantized_model_path.glob("*.xml"))
        bin_files = list(quantized_model_path.glob("*.bin"))
        
        if not xml_files or not bin_files:
            raise FileNotFoundError(
                f"OpenVINO model files (.xml/.bin) not found in: {quantized_model_path}"
            )
        
        # Copy .xml file as model.xml
        xml_src = xml_files[0]
        xml_dst = version_dir / "model.xml"
        shutil.copy2(xml_src, xml_dst)
        logger.info(f"Copied {xml_src.name} to {xml_dst}")
        
        # Copy .bin file as model.bin
        bin_src = bin_files[0]
        bin_dst = version_dir / "model.bin"
        shutil.copy2(bin_src, bin_dst)
        logger.info(f"Copied {bin_src.name} to {bin_dst}")
        
        logger.info(f"OpenVINO model files copied to: {version_dir}")
    else:
        # ONNX and TensorRT are single files
        # Determine destination filename based on platform
        if quantization_type == "onnx":
            dst_filename = "model.onnx"
        elif quantization_type == "tensorrt":
            dst_filename = "model.plan"
        else:
            dst_filename = quantized_model_path.name
        
        dst_path = version_dir / dst_filename
        
        logger.info(f"Copying {quantization_type} model to: {dst_path}")
        
        # Remove existing model file if exists
        if dst_path.exists():
            dst_path.unlink()
        
        # Copy model file
        shutil.copy2(quantized_model_path, dst_path)
        logger.info(f"Model copied to: {dst_path}")
    
    # Create config.pbtxt with dynamic batching enabled
    config_path = model_dir / "config.pbtxt"
    config_content = create_triton_config(
        model_name=model_name,
        platform=platform,
        quantization_type=quantization_type,
        max_batch_size=MAX_BATCH,
        enable_dynamic_batching=True,  # Always enable dynamic batching
    )
    
    logger.info(f"Writing Triton config to: {config_path}")
    with open(config_path, "w") as f:
        f.write(config_content)
    
    logger.info(f"Triton model repository setup completed: {model_dir}")
    return str(model_dir)


def auto_setup_triton(
    quantized_model_path: Optional[str] = None,
    quantization_type: Optional[str] = None,
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    auto_quantize: bool = True,
    **kwargs
) -> str:
    """
    Automatically setup Triton model repository.
    If quantized_model_path is not provided, it will try to find it.
    If model is not found and auto_quantize is True, it will run quantization automatically.
    
    Args:
        quantized_model_path: Path to quantized model (if already exists)
        quantization_type: Type of quantization
        model_path: Original model path (used to find quantized model)
        output_dir: Directory where quantized model was saved
        auto_quantize: Automatically run quantization if model not found
        **kwargs: Additional arguments
        
    Returns:
        Path to model directory in Triton repository
        
    Raises:
        FileNotFoundError: If model is not found and auto_quantize is False
        RuntimeError: If setup fails
    """
    quantization_type = (quantization_type or QUANTIZATION_TYPE).lower()
    
    logger.info(f"Starting auto Triton setup with quantization type: {quantization_type}")
    
    # Find quantized model if not provided
    if quantized_model_path is None:
        try:
            quantized_model_path = find_quantized_model(
                quantization_type=quantization_type,
                model_path=model_path,
                output_dir=output_dir,
            )
            logger.info(f"Found existing quantized model: {quantized_model_path}")
        except FileNotFoundError:
            if auto_quantize:
                logger.info("Quantized model not found, running quantization automatically...")
                try:
                    from src.quantization.auto_quantization import auto_quantize as quantize_fn
                    quantized_model_path = quantize_fn(
                        quantization_type=quantization_type,
                        model_path=model_path,
                        output_dir=output_dir,
                        **{k: v for k, v in kwargs.items() if k not in ["model_name", "model_version", "repository_dir"]}
                    )
                    logger.info(f"Quantization completed: {quantized_model_path}")
                except ImportError as e:
                    logger.error(f"Failed to import quantization module: {e}")
                    raise RuntimeError("Cannot import quantization module. Please check installation.")
            else:
                raise
    else:
        quantized_model_path = Path(quantized_model_path)
        if not quantized_model_path.exists():
            raise FileNotFoundError(f"Quantized model not found: {quantized_model_path}")
    
    # Setup repository
    model_dir = setup_triton_model_repository(
        quantized_model_path=quantized_model_path,
        quantization_type=quantization_type,
        model_name=kwargs.get("model_name"),
        model_version=kwargs.get("model_version"),
        repository_dir=kwargs.get("repository_dir"),
    )
    
    return model_dir


def quantize_and_setup_triton(
    quantization_type: Optional[str] = None,
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    force_requantize: bool = False,
    **kwargs
) -> str:
    """
    Automatically quantize model and setup Triton repository.
    This is the main function that combines quantization and Triton setup.
    
    Args:
        quantization_type: Type of quantization ('onnx', 'tensorrt', 'openvino')
        model_path: Path to original model weights (.pt file)
        output_dir: Directory to save quantized model
        force_requantize: Force re-quantization even if model exists
        **kwargs: Additional arguments for quantization and Triton setup
        
    Returns:
        Path to model directory in Triton repository
        
    Raises:
        FileNotFoundError: If model files don't exist
        RuntimeError: If quantization or setup fails
    """
    quantization_type = (quantization_type or QUANTIZATION_TYPE).lower()
    model_path = model_path or MODEL_WEIGHTS_PATH
    
    logger.info(f"Starting quantize and setup Triton with type: {quantization_type}")
    
    # Try to import auto_quantize
    try:
        from src.quantization.auto_quantization import auto_quantize
    except ImportError as e:
        logger.error(f"Failed to import auto_quantize: {e}")
        raise RuntimeError("Cannot import quantization module. Please check installation.")
    
    # Check if quantized model already exists
    quantized_model_path = None
    if not force_requantize:
        try:
            quantized_model_path = find_quantized_model(
                quantization_type=quantization_type,
                model_path=model_path,
                output_dir=output_dir,
            )
            logger.info(f"Found existing quantized model: {quantized_model_path}")
        except FileNotFoundError:
            logger.info("Quantized model not found, will run quantization...")
    
    # Run quantization if needed
    if quantized_model_path is None or force_requantize:
        logger.info("Running quantization...")
        quantized_model_path = auto_quantize(
            quantization_type=quantization_type,
            model_path=model_path,
            output_dir=output_dir,
            **{k: v for k, v in kwargs.items() if k not in ["model_name", "model_version", "repository_dir"]}
        )
        logger.info(f"Quantization completed: {quantized_model_path}")
    
    # Setup Triton repository
    logger.info("Setting up Triton repository...")
    model_repository_path = auto_setup_triton(
        quantized_model_path=str(quantized_model_path),
        quantization_type=quantization_type,
        model_path=model_path,
        output_dir=output_dir,
        **{k: v for k, v in kwargs.items() if k in ["model_name", "model_version", "repository_dir"]}
    )
    
    logger.info(f"Quantization and Triton setup completed: {model_repository_path}")
    return model_repository_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto setup Triton Inference Server")
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Run quantization before setting up Triton"
    )
    parser.add_argument(
        "--force-quantize",
        action="store_true",
        help="Force re-quantization even if model exists"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to original model weights (.pt file)"
    )
    parser.add_argument(
        "--quantization-type",
        type=str,
        choices=["onnx", "tensorrt", "openvino"],
        help="Type of quantization"
    )
    
    args = parser.parse_args()
    
    try:
        if args.quantize or args.force_quantize:
            # Run quantization and setup Triton
            model_repository_path = quantize_and_setup_triton(
                quantization_type=args.quantization_type,
                model_path=args.model_path,
                force_requantize=args.force_quantize,
            )
            logger.info(f"Quantization and Triton setup completed: {model_repository_path}")
        else:
            # Only setup Triton (will auto-quantize if model not found)
            model_repository_path = auto_setup_triton(
                quantization_type=args.quantization_type,
                model_path=args.model_path,
                auto_quantize=True,  # Auto-quantize if model not found
            )
            logger.info(f"Triton repository setup completed: {model_repository_path}")
    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=True)
        sys.exit(1)

