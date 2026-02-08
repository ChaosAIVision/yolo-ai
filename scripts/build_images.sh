#!/bin/bash
# Script to build complete Docker images once
# This builds images with all dependencies pre-installed

set -e

echo "=========================================="
echo "Building Complete Docker Images"
echo "=========================================="
echo "This will build images with all dependencies."
echo "Images will be cached for future use."
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Build quantization image
echo "Building quantization image (this may take a while)..."
echo "Installing: torch 2.8.0 (ROCm 6.4), ultralytics, and all dependencies"
docker compose --profile quantization build --no-cache quantization

echo ""
echo "=========================================="
echo "Image build completed!"
echo "=========================================="
echo "Quantization image: yolo-ai-quantization:latest"
echo ""
echo "You can now run: ./scripts/deploy.sh"
echo "The image will be reused (no rebuild needed)."
echo "=========================================="

