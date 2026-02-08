#!/bin/bash
# Test script for Triton Inference Server

set -e

TRITON_URL="${TRITON_URL:-http://localhost:7000}"
MODEL_NAME="${MODEL_NAME:-yolov8n}"

echo "=========================================="
echo "Testing Triton Inference Server"
echo "=========================================="
echo "Triton URL: $TRITON_URL"
echo "Model: $MODEL_NAME"
echo ""

# Test 1: Health check
echo "1. Health Check..."
if curl -s -f "${TRITON_URL}/v2/health/ready" > /dev/null; then
    echo "   ✓ Server is ready"
else
    echo "   ✗ Server is not ready"
    exit 1
fi

# Test 2: Server metadata
echo ""
echo "2. Server Metadata..."
SERVER_INFO=$(curl -s "${TRITON_URL}/v2")
echo "   ✓ Server version: $(echo $SERVER_INFO | python3 -c "import sys, json; print(json.load(sys.stdin)['version'])" 2>/dev/null || echo 'N/A')"

# Test 3: Model metadata
echo ""
echo "3. Model Metadata..."
MODEL_INFO=$(curl -s "${TRITON_URL}/v2/models/${MODEL_NAME}")
MODEL_VERSION=$(echo $MODEL_INFO | python3 -c "import sys, json; print(json.load(sys.stdin)['versions'][0])" 2>/dev/null || echo 'N/A')
echo "   ✓ Model version: $MODEL_VERSION"

# Test 4: Model config
echo ""
echo "4. Model Config..."
CONFIG=$(curl -s "${TRITON_URL}/v2/models/${MODEL_NAME}/config")
MAX_BATCH=$(echo $CONFIG | python3 -c "import sys, json; print(json.load(sys.stdin).get('max_batch_size', 'N/A'))" 2>/dev/null || echo 'N/A')
echo "   ✓ Max batch size: $MAX_BATCH"

# Test 5: Inference test
echo ""
echo "5. Inference Test..."
python3 << EOF
import json
import numpy as np
import requests
import sys

try:
    # Create dummy input (1, 3, 640, 640) FP32
    input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
    
    payload = {
        "inputs": [
            {
                "name": "images",
                "shape": [1, 3, 640, 640],
                "datatype": "FP32",
                "data": input_data.tolist()
            }
        ],
        "outputs": [{"name": "output0"}]
    }
    
    response = requests.post(
        "${TRITON_URL}/v2/models/${MODEL_NAME}/infer",
        json=payload,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        output_shape = result['outputs'][0]['shape']
        print(f"   ✓ Inference successful!")
        print(f"   ✓ Output shape: {output_shape}")
        print(f"   ✓ Output datatype: {result['outputs'][0]['datatype']}")
    else:
        print(f"   ✗ Inference failed: {response.status_code}")
        print(f"   Response: {response.text}")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)
EOF

echo ""
echo "=========================================="
echo "✓ All tests passed!"
echo "=========================================="

