#!/bin/bash
# Convert YOLO model to ONNX using conda yolo_env

set -e

CONDA_ENV="yolo_env"
MODEL_PATH="${1:-/home/clara/manhhd/yolo_ppe/yolo-ai/weights/yolov8n.pt}"
OUTPUT_DIR="${2:-/home/clara/manhhd/yolo_ppe/yolo-ai/weights}"

echo "Activating conda environment: $CONDA_ENV"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "Converting model to ONNX: $MODEL_PATH"
python -m src.quantization.onnx

echo "ONNX conversion completed. Model saved to: $OUTPUT_DIR"

