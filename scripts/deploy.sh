#!/bin/bash
# Main deployment script
# STEP 1: Auto quantization + tạo model repository (phải hoàn thành trước)
# STEP 2: Start services (chỉ chạy sau khi STEP 1 hoàn thành)

set -e

echo "=========================================="
echo "YOLO AI Deployment Script"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Check if weights directory exists
WEIGHTS_DIR="$PROJECT_ROOT/weights"
if [ ! -d "$WEIGHTS_DIR" ]; then
    echo "Creating weights directory..."
    mkdir -p "$WEIGHTS_DIR"
fi

# Check if model files exist
MODEL_FILES=($(find "$WEIGHTS_DIR" -name "*.pt" -type f 2>/dev/null))

if [ ${#MODEL_FILES[@]} -eq 0 ]; then
    echo "=========================================="
    echo "ERROR: No .pt model files found in $WEIGHTS_DIR"
    echo "Please place your model file (.pt) in the weights directory"
    echo "=========================================="
    exit 1
fi

echo "Found ${#MODEL_FILES[@]} model file(s):"
for model in "${MODEL_FILES[@]}"; do
    echo "  - $model"
done

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set default quantization type
export QUANTIZATION_TYPE=${QUANTIZATION_TYPE:-onnx}
echo "Quantization type: $QUANTIZATION_TYPE"

# ==========================================
# STEP 1: Auto quantization + Model repository
# ==========================================
echo ""
echo "=========================================="
echo "STEP 1: Auto Quantization + Model Repository"
echo "=========================================="
echo "This step MUST complete before proceeding to STEP 2"
echo ""

# Stop existing quantization container if any
echo "Stopping existing quantization containers..."
docker compose --profile quantization down || true

# Check if quantization image exists
QUANTIZATION_IMAGE=$(docker images -q yolo-ai-quantization:latest 2>/dev/null || echo "")

if [ -z "$QUANTIZATION_IMAGE" ]; then
    echo ""
    echo "⚠ Quantization image not found!"
    echo "Please build the image first by running:"
    echo "  ./scripts/build_images.sh"
    echo ""
    echo "Or build manually:"
    echo "  docker compose --profile quantization build quantization"
    echo ""
    exit 1
else
    echo "✓ Quantization image found. Using cached image (no rebuild needed)."
fi

# Run quantization (one-time service)
echo "Running quantization in container..."
if docker compose --profile quantization run --rm quantization; then
    echo ""
    echo "✓ STEP 1 completed successfully!"
    echo ""
else
    echo ""
    echo "✗ STEP 1 FAILED!"
    echo "  Please check the logs above for errors"
    echo "  Cannot proceed to STEP 2"
    exit 1
fi

# Display STEP 1 results
echo "=========================================="
echo "STEP 1 Results:"
echo "=========================================="

# Check quantized model
QUANTIZED_MODEL=""
if [ "$QUANTIZATION_TYPE" = "onnx" ]; then
    QUANTIZED_MODEL=$(find "$WEIGHTS_DIR" -name "*.onnx" -type f 2>/dev/null | head -n 1)
elif [ "$QUANTIZATION_TYPE" = "tensorrt" ]; then
    QUANTIZED_MODEL=$(find "$WEIGHTS_DIR" -name "*.engine" -type f 2>/dev/null | head -n 1)
elif [ "$QUANTIZATION_TYPE" = "openvino" ]; then
    QUANTIZED_MODEL=$(find "$WEIGHTS_DIR" -name "*_openvino" -type d 2>/dev/null | head -n 1)
fi

if [ -n "$QUANTIZED_MODEL" ]; then
    echo "✓ Quantized model created:"
    echo "  Location: $QUANTIZED_MODEL"
    if [ -f "$QUANTIZED_MODEL" ]; then
        SIZE=$(du -h "$QUANTIZED_MODEL" | cut -f1)
        echo "  Size: $SIZE"
    elif [ -d "$QUANTIZED_MODEL" ]; then
        echo "  Type: Directory"
    fi
else
    echo "⚠ Quantized model not found (may be in different location)"
fi

# Verify model repository was created
MODEL_REPO="$PROJECT_ROOT/src/triton_config/model_repository"
if [ ! -d "$MODEL_REPO" ] || [ -z "$(ls -A $MODEL_REPO 2>/dev/null)" ]; then
    echo "✗ ERROR: Model repository not found or empty: $MODEL_REPO"
    exit 1
fi

echo ""
echo "✓ Model repository created:"
echo "  Location: $MODEL_REPO"
echo "  Contents:"
for model_dir in "$MODEL_REPO"/*; do
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
                        FILE_SIZE=$(du -h "$model_file" | cut -f1)
                        echo "        - $FILE_NAME ($FILE_SIZE)"
                    fi
                done
            fi
        done
    fi
done

echo ""
echo "=========================================="
echo ""

# ==========================================
# STEP 2: Start Services
# ==========================================
echo "=========================================="
echo "STEP 2: Starting Services"
echo "=========================================="
echo "Starting Triton and Backend services..."
echo ""

# Stop existing services
echo "Stopping existing services..."
docker compose down || true

# Build service images
echo "Building service images..."
docker compose build triton-yolo backend

# Start services
echo "Starting services..."
docker compose up -d triton-yolo backend

# Wait for services to be ready
echo ""
echo "Waiting for services to be ready..."

# Wait for Triton
echo "  - Waiting for Triton server..."
timeout=60
elapsed=0
while ! curl -f http://localhost:7000/v2/health/ready > /dev/null 2>&1; do
    if [ $elapsed -ge $timeout ]; then
        echo "    ✗ Triton server did not become ready"
        docker compose logs triton-yolo
        exit 1
    fi
    echo "    Waiting... ($elapsed/$timeout seconds)"
    sleep 2
    elapsed=$((elapsed + 2))
done
echo "    ✓ Triton server is ready"

# Wait for Backend API
echo "  - Waiting for Backend API..."
timeout=60
elapsed=0
while ! curl -f http://localhost:8000/health > /dev/null 2>&1; do
    if [ $elapsed -ge $timeout ]; then
        echo "    ⚠ Backend API did not become ready (checking logs...)"
        docker compose logs backend | tail -20
        break
    fi
    echo "    Waiting... ($elapsed/$timeout seconds)"
    sleep 2
    elapsed=$((elapsed + 2))
done

# Wait for BentoML (optional check)
echo "  - Waiting for BentoML API..."
timeout=60
elapsed=0
while ! curl -f http://localhost:3000/healthz > /dev/null 2>&1; do
    if [ $elapsed -ge $timeout ]; then
        echo "    ⚠ BentoML API did not become ready (checking logs...)"
        docker compose logs backend | tail -20
        break
    fi
    echo "    Waiting... ($elapsed/$timeout seconds)"
    sleep 2
    elapsed=$((elapsed + 2))
done

# Final status
echo ""
echo "=========================================="
echo "Deployment completed!"
echo "=========================================="
echo "Services:"
echo "  - Triton Server:    http://localhost:7000"
echo "  - BentoML API:      http://localhost:3000"
echo "  - Backend API:      http://localhost:8000"
echo ""
echo "Commands:"
echo "  View logs:          docker compose logs -f"
echo "  Stop services:      docker compose down"
echo "  Restart services:   docker compose restart"
echo "=========================================="
