#!/bin/bash
# Script to check quantization results and checkpoints

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Quantization Results Check"
echo "=========================================="
echo ""

# Check quantized models
echo "1. Quantized Models (weights/):"
echo "----------------------------------------"
if [ -d "$PROJECT_ROOT/weights" ]; then
    find "$PROJECT_ROOT/weights" -type f \( -name "*.onnx" -o -name "*.engine" -o -name "*.bin" -o -name "*.xml" \) -exec ls -lh {} \; 2>/dev/null | while read line; do
        echo "  $line"
    done
    
    if [ -z "$(find "$PROJECT_ROOT/weights" -type f \( -name "*.onnx" -o -name "*.engine" -o -name "*.bin" \) 2>/dev/null)" ]; then
        echo "  ⚠ No quantized models found"
    fi
else
    echo "  ⚠ weights/ directory not found"
fi

echo ""

# Check model repository
echo "2. Triton Model Repository:"
echo "----------------------------------------"
MODEL_REPO="$PROJECT_ROOT/src/triton_config/model_repository"
if [ -d "$MODEL_REPO" ]; then
    for model_dir in "$MODEL_REPO"/*; do
        if [ -d "$model_dir" ]; then
            MODEL_NAME=$(basename "$model_dir")
            echo "  Model: $MODEL_NAME"
            
            # Check config
            if [ -f "$model_dir/config.pbtxt" ]; then
                echo "    ✓ config.pbtxt exists"
                # Show platform
                PLATFORM=$(grep "platform:" "$model_dir/config.pbtxt" | awk '{print $2}' | tr -d '"')
                echo "    Platform: $PLATFORM"
            else
                echo "    ✗ config.pbtxt not found"
            fi
            
            # Check versions
            for version_dir in "$model_dir"/*; do
                if [ -d "$version_dir" ]; then
                    VERSION=$(basename "$version_dir")
                    echo "    Version: $VERSION"
                    for model_file in "$version_dir"/*; do
                        if [ -f "$model_file" ]; then
                            FILE_NAME=$(basename "$model_file")
                            FILE_SIZE=$(du -h "$model_file" | cut -f1)
                            echo "      ✓ $FILE_NAME ($FILE_SIZE)"
                        fi
                    done
                fi
            done
            echo ""
        fi
    done
else
    echo "  ⚠ Model repository not found: $MODEL_REPO"
fi

echo ""

# Check BentoML checkpoint
echo "3. BentoML Checkpoint (bentoml/):"
echo "----------------------------------------"
if [ -d "$PROJECT_ROOT/bentoml" ]; then
    BENTO_COUNT=$(find "$PROJECT_ROOT/bentoml" -type d -name "*.bento" 2>/dev/null | wc -l)
    if [ "$BENTO_COUNT" -gt 0 ]; then
        echo "  ✓ Found $BENTO_COUNT bento(s):"
        find "$PROJECT_ROOT/bentoml" -type d -name "*.bento" 2>/dev/null | while read bento; do
            BENTO_NAME=$(basename "$bento")
            BENTO_SIZE=$(du -sh "$bento" 2>/dev/null | cut -f1)
            echo "    - $BENTO_NAME ($BENTO_SIZE)"
        done
    else
        echo "  ⚠ No bento found (BentoML hasn't been built yet)"
    fi
else
    echo "  ⚠ bentoml/ directory not found"
fi

echo ""

# Summary
echo "=========================================="
echo "Summary:"
echo "=========================================="
ONNX_COUNT=$(find "$PROJECT_ROOT/weights" -name "*.onnx" -type f 2>/dev/null | wc -l)
ENGINE_COUNT=$(find "$PROJECT_ROOT/weights" -name "*.engine" -type f 2>/dev/null | wc -l)
REPO_COUNT=$(find "$MODEL_REPO" -name "config.pbtxt" -type f 2>/dev/null | wc -l)

echo "  Quantized models: $ONNX_COUNT ONNX, $ENGINE_COUNT TensorRT"
echo "  Model repositories: $REPO_COUNT"
echo ""
echo "Locations:"
echo "  - Quantized models: $PROJECT_ROOT/weights/"
echo "  - Model repository: $MODEL_REPO"
echo "  - BentoML data: $PROJECT_ROOT/bentoml/"
echo "=========================================="

