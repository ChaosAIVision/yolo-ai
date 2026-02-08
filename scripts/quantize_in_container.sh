#!/bin/bash
# Script to run quantization and setup Triton repository inside container
set -e

echo "=========================================="
echo "Starting quantization and Triton setup"
echo "=========================================="

# Set default paths (mounted volumes)
WEIGHTS_DIR="/workspace/weights"
MODEL_REPOSITORY="/workspace/model_repository"
PROJECT_ROOT="/workspace"

# Check if weights directory exists and has model files
if [ ! -d "$WEIGHTS_DIR" ]; then
    echo "ERROR: Weights directory not found: $WEIGHTS_DIR"
    exit 1
fi

# Find .pt model files
MODEL_FILES=($(find "$WEIGHTS_DIR" -name "*.pt" -type f))

if [ ${#MODEL_FILES[@]} -eq 0 ]; then
    echo "ERROR: No .pt model files found in $WEIGHTS_DIR"
    exit 1
fi

# Use first model file found
MODEL_PATH="${MODEL_FILES[0]}"
echo "Found model: $MODEL_PATH"

# Get model name
MODEL_NAME=$(basename "$MODEL_PATH" .pt)
echo "Model name: $MODEL_NAME"

# Set environment variables
export MODEL_WEIGHTS_PATH="$MODEL_PATH"
export ONNX_OUTPUT_DIR="$WEIGHTS_DIR"
export QUANTIZATION_TYPE="${QUANTIZATION_TYPE:-onnx}"

echo "Quantization type: $QUANTIZATION_TYPE"
echo "Output directory: $ONNX_OUTPUT_DIR"

# Change to project root
cd "$PROJECT_ROOT"

# Run quantization and setup Triton
echo "Running quantization and Triton setup..."
python3 -m src.triton.auto_config --quantize

echo ""
echo "=========================================="
echo "Quantization and Triton setup completed"
echo "=========================================="
echo ""

# Create summary
echo "Creating quantization summary..."
if [ -f "$PROJECT_ROOT/scripts/create_quantization_summary.sh" ]; then
    bash "$PROJECT_ROOT/scripts/create_quantization_summary.sh" || echo "Warning: Failed to create summary"
fi

echo ""
echo "Results:"
echo "  - Quantized model saved to: $WEIGHTS_DIR"
echo "  - Model repository created at: $MODEL_REPOSITORY"
echo "  - Summary file: $PROJECT_ROOT/quantization_summary.json"
echo ""
echo "Files created:"
echo "  Quantized model:"
if [ "$QUANTIZATION_TYPE" = "onnx" ]; then
    ONNX_FILES=$(find "$WEIGHTS_DIR" -name "*.onnx" -type f 2>/dev/null)
    for file in $ONNX_FILES; do
        if [ -f "$file" ]; then
            SIZE=$(du -h "$file" 2>/dev/null | cut -f1)
            echo "    - $(basename $file) ($SIZE)"
        fi
    done
elif [ "$QUANTIZATION_TYPE" = "tensorrt" ]; then
    ENGINE_FILES=$(find "$WEIGHTS_DIR" -name "*.engine" -type f 2>/dev/null)
    for file in $ENGINE_FILES; do
        if [ -f "$file" ]; then
            SIZE=$(du -h "$file" 2>/dev/null | cut -f1)
            echo "    - $(basename $file) ($SIZE)"
        fi
    done
elif [ "$QUANTIZATION_TYPE" = "openvino" ]; then
    OPENVINO_DIRS=$(find "$WEIGHTS_DIR" -name "*_openvino" -type d 2>/dev/null)
    for dir in $OPENVINO_DIRS; do
        if [ -d "$dir" ]; then
            echo "    - $(basename $dir)/"
            for file in "$dir"/*.xml "$dir"/*.bin; do
                if [ -f "$file" ]; then
                    SIZE=$(du -h "$file" 2>/dev/null | cut -f1)
                    echo "      - $(basename $file) ($SIZE)"
                fi
            done
        fi
    done
fi

echo ""
echo "  Model repository:"
if [ -d "$MODEL_REPOSITORY" ]; then
    for model_dir in "$MODEL_REPOSITORY"/*; do
        if [ -d "$model_dir" ]; then
            MODEL_NAME=$(basename "$model_dir")
            echo "    - $MODEL_NAME/"
            if [ -f "$model_dir/config.pbtxt" ]; then
                echo "      - config.pbtxt"
            fi
            for version_dir in "$model_dir"/*; do
                if [ -d "$version_dir" ]; then
                    VERSION=$(basename "$version_dir")
                    echo "      - $VERSION/"
                    for model_file in "$version_dir"/*; do
                        if [ -f "$model_file" ]; then
                            FILE_NAME=$(basename "$model_file")
                            FILE_SIZE=$(du -h "$model_file" 2>/dev/null | cut -f1)
                            echo "        - $FILE_NAME ($FILE_SIZE)"
                        fi
                    done
                fi
            done
        fi
    done
fi
echo ""

