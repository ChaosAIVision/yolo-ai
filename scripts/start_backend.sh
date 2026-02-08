#!/bin/bash
# Script to start backend services using CLI commands
# - BentoML: bentoml serve (CLI)
# - API: python -m src.api.v1 (CLI)

set -e

echo "=========================================="
echo "Starting backend services"
echo "=========================================="

cd /app

# Wait for Triton to be ready
echo "Waiting for Triton server to be ready..."
timeout=60
elapsed=0
while ! curl -f http://triton-yolo:8000/v2/health/ready > /dev/null 2>&1; do
    if [ $elapsed -ge $timeout ]; then
        echo "ERROR: Triton server did not become ready within $timeout seconds"
        exit 1
    fi
    echo "Waiting for Triton server... ($elapsed/$timeout seconds)"
    sleep 2
    elapsed=$((elapsed + 2))
done
echo "✓ Triton server is ready!"

# Function to handle cleanup
cleanup() {
    echo "Shutting down services..."
    kill $BENTOML_PID $API_PID ${UI_PID:-} 2>/dev/null || true
    exit 0
}

trap cleanup SIGTERM SIGINT

# Step 1: Start BentoML service using CLI
echo "Starting BentoML service on port 3000 (CLI)..."
bentoml serve src.deploy.service:svc --port 3000 --host 0.0.0.0 > /var/log/bentoml.log 2>&1 &
BENTOML_PID=$!

# Wait for BentoML to start
echo "Waiting for BentoML service to start..."
sleep 10

# Check if BentoML is running
if ! kill -0 $BENTOML_PID 2>/dev/null; then
    echo "ERROR: BentoML service failed to start"
    cat /var/log/bentoml.log
    exit 1
fi
echo "✓ BentoML service started (PID: $BENTOML_PID)"

# Step 2: Start API server using CLI
echo "Starting API server on port 5000 (CLI)..."
python3 -m src.api.v1 --host 0.0.0.0 --port 5000 > /var/log/api.log 2>&1 &
API_PID=$!

# Wait for API server to start
echo "Waiting for API server to start..."
sleep 5

# Check if API is running
if ! kill -0 $API_PID 2>/dev/null; then
    echo "ERROR: API server failed to start"
    cat /var/log/api.log
    exit 1
fi
echo "✓ API server started (PID: $API_PID)"

# Step 3: Start UI frontend (if app directory exists)
if [ -d "/app/app" ] && [ -f "/app/app/package.json" ]; then
    echo "Starting UI frontend on port 8080 (npm run dev)..."
    cd /app/app
    npm run dev -- --host 0.0.0.0 --port 8080 > /var/log/ui.log 2>&1 &
    UI_PID=$!
    cd /app
    
    # Wait for UI to start
    echo "Waiting for UI frontend to start..."
    sleep 5
    
    # Check if UI is running
    if ! kill -0 $UI_PID 2>/dev/null; then
        echo "WARNING: UI frontend failed to start"
        cat /var/log/ui.log
    else
        echo "✓ UI frontend started (PID: $UI_PID)"
    fi
else
    echo "⚠ UI frontend not found (app/package.json missing)"
    UI_PID=""
fi

echo ""
echo "=========================================="
echo "All services started successfully"
echo "=========================================="
echo "BentoML API: http://localhost:3000"
echo "Backend API: http://localhost:5000"
echo "UI Frontend: http://localhost:7070"
echo "Triton Server: http://localhost:7000"
echo "=========================================="

# Keep script running and monitor processes
while true; do
    if ! kill -0 $BENTOML_PID 2>/dev/null; then
        echo "ERROR: BentoML process died"
        cat /var/log/bentoml.log
        exit 1
    fi
    if ! kill -0 $API_PID 2>/dev/null; then
        echo "ERROR: API process died"
        cat /var/log/api.log
        exit 1
    fi
    if [ -n "$UI_PID" ] && ! kill -0 $UI_PID 2>/dev/null; then
        echo "ERROR: UI process died"
        cat /var/log/ui.log
        exit 1
    fi
    sleep 5
done
