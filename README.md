# YOLO-AI: High-Performance Real-Time Object Detection Framework


---

## Overview

YOLO-AI is a complete end-to-end framework for real-time object detection that enables seamless deployment from model training to production. Built with YOLOv8, BentoML, and WebSocket streaming, it delivers high-quality detection results with minimal latency.

---

## Complete Workflow: Train → Convert → Deploy

### Workflow Overview

| Stage | Description | Output |
|-------|-------------|--------|
| **1. Train** | Train YOLOv8 model on custom dataset | `.pt` model weights |
| **2. Convert** | Convert PyTorch model to ONNX format | `.onnx` optimized model |
| **3. Deploy** | Deploy ONNX model to BentoML service | Production-ready API service |

### Detailed Steps

#### 1. Train Model
```bash
# Train YOLOv8 model on your dataset
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='your_dataset.yaml', epochs=100, imgsz=640)
# Output: weights/best.pt
```

#### 2. Convert to ONNX
```bash
# Convert trained model to ONNX for optimized inference
python -m src.quantization.onnx_model \
  --model_path weights/best.pt \
  --output_path weights/
```

#### 3. Deploy to BentoML
```bash
# Deploy ONNX model to BentoML
python -m src.deploy.deploy \
  --onnx_path weights/best.onnx

# Build BentoML service
bentoml build

# Serve locally
bentoml serve yolov8-service:latest --port 3000
```

---

## Application Flow: UI → Backend → UI

### Request Flow Diagram

```
┌─────────────┐
│   Frontend  │
│   (React)   │
└──────┬──────┘
       │
       │ 1. User Action (Upload/Stream)
       │
       ▼
┌─────────────────────────────────────┐
│      API Server (aiohttp)           │
│      Port: 8005                      │
└──────┬───────────────────┬───────────┘
       │                   │
       │ 2. Process        │ 3. WebSocket
       │    Request        │    Stream
       │                   │
       ▼                   ▼
┌──────────────┐    ┌─────────────────┐
│ BentoML      │    │ YouTube Stream  │
│ Service      │    │ (yt-dlp+ffmpeg) │
│ Port: 3000   │    │                 │
└──────┬───────┘    └────────┬────────┘
       │                     │
       │ 4. YOLO Inference  │ 5. Frame Processing
       │    (ONNX Runtime)   │    (YOLO Detection)
       │                     │
       └──────────┬──────────┘
                  │
                  │ 6. Annotated Frame
                  │
                  ▼
         ┌─────────────────┐
         │  WebSocket      │
         │  Response       │
         └────────┬────────┘
                  │
                  │ 7. Display Result
                  │
                  ▼
         ┌─────────────────┐
         │   Frontend UI   │
         │   (Canvas)       │
         └─────────────────┘
```

### Step-by-Step Flow

| Step | Component | Action | Data Flow |
|------|-----------|--------|-----------|
| **1** | Frontend | User uploads image/YouTube URL | Image/URL → API Server |
| **2** | API Server | Receives request, processes frame | Frame → BentoML Service |
| **3** | BentoML | YOLO inference on frame | Frame → Detections |
| **4** | API Server | Annotates frame with bounding boxes | Detections → Annotated Frame |
| **5** | WebSocket | Streams annotated frames | Annotated Frame → Frontend |
| **6** | Frontend | Displays result on canvas | Annotated Frame → UI Display |

---

## Frontend Features

### Feature Overview

| Feature | Description | Component |
|---------|-------------|-----------|
| **Image Upload** | Upload single image for detection | `ImageUpload.tsx` |
| **YouTube Streaming** | Stream YouTube videos with real-time detection | `VideoStreamUpload.tsx` |
| **IP Camera** | Connect to IP cameras for live detection | `IPCameraStream.tsx` |

### Detailed Features

#### 1. Image Upload & Detection
- **Drag-and-drop** image upload interface
- **Real-time** annotation with bounding boxes
- **Download** annotated results
- **Support formats**: JPEG, PNG
- **Display**: Confidence scores and class labels

#### 2. YouTube Video Streaming
- **URL input** for YouTube videos
- **Real-time streaming** via WebSocket
- **Frame-by-frame** detection processing
- **FPS counter** and detection statistics
- **Play/Stop** controls

#### 3. IP Camera Streaming
- **IP camera** connection support
- **Local device** camera access
- **Live streaming** with real-time detection
- **Connection status** indicator

---

## Backend Features

### Core Capabilities

| Feature | Technology | Description |
|---------|------------|-------------|
| **Model Serving** | BentoML | Production-ready ML model serving |
| **Real-time Streaming** | WebSocket | Low-latency video streaming |
| **Video Processing** | yt-dlp + ffmpeg | YouTube stream extraction and decoding |
| **Object Detection** | YOLOv8 + ONNX | High-performance inference |
| **Image Processing** | OpenCV + PIL | Frame annotation and encoding |

### API Endpoints

| Endpoint | Method | Description | Input | Output |
|----------|--------|-------------|-------|--------|
| `/api/v1/upload` | POST | Upload image for detection | Image file | Annotated JPEG |
| `/ws/youtube` | WebSocket | YouTube video streaming | YouTube URL | Annotated frames (base64) |


--


## Example: PPE (Personal Protective Equipment) Detection

### Use Case
Detect Personal Protective Equipment (PPE) including:
- **Person**
- **Helmet**
- **Vest**
- **Shoes**

### Detection Results

![PPE Detection Result](assert/result_detect.jpeg)

### Configuration

```python
CLASS_NAMES = {
    0: "person",
    1: "helmet",
    2: "vest",
    3: "shoes"
}
```

### Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **FPS** | 18-20 | Frames per second processed |
| **Latency** | <100ms | End-to-end detection time |
| **Accuracy** | High | YOLOv8-based detection |
| **Frame Skip** | Every 3rd frame | Optimized processing |

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.10+ | Backend runtime |
| Node.js | 18+ | Frontend runtime |
| CUDA | 11.8+ (optional) | GPU acceleration |
| ffmpeg | Latest | Video processing |
| yt-dlp | Latest | YouTube extraction |

### Installation

1. **Clone repository**
   ```bash
   git clone <repository-url>
   cd yolo-ai
   ```

2. **Install backend dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**
   ```bash
   cd app
   npm install
   ```

### Running the Application

#### 1. Start BentoML Service
```bash
bentoml serve yolov8-service:latest --port 3000
```

#### 2. Start API Server
```bash
python -m src.api.v1 --host 0.0.0.0 --port 8005
```

#### 3. Start Frontend
```bash
cd app
npm run dev
```

#### 4. Access Application
- **Frontend**: `http://localhost:8081`
- **API Server**: `http://localhost:8005`
- **BentoML Service**: `http://localhost:3000`

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENTO_ENDPOINT_URL` | `http://localhost:3000` | BentoML service URL |
| `FPS_LIMIT` | `20` | Maximum frames per second |
| `FRAME_SKIP` | `3` | Process every Nth frame |
| `CONF_THRES` | `0.20` | Confidence threshold |
| `IOU_THRES` | `0.3` | IoU threshold for NMS |
| `CUDA_VISIBLE_DEVICES` | `1` | GPU device ID |

---

## 📦 Project Structure

```
yolo-ai/
├── src/
│   ├── api/              # API endpoints (WebSocket, REST)
│   ├── deploy/           # BentoML deployment
│   ├── quantization/     # Model conversion (ONNX, TensorRT)
│   └── config.py        # Configuration
├── app/                  # Frontend (React + TypeScript)
│   ├── src/
│   │   ├── components/  # UI components
│   │   ├── hooks/       # React hooks
│   │   └── lib/         # Utilities
├── weights/              # Model weights
├── scripts/              # Utility scripts
└── requirements.txt     # Python dependencies
```

---

## 🎯 Key Advantages

| Advantage | Description |
|-----------|-------------|
| **Easy Deployment** | One-command deployment with BentoML |
| **High Quality** | YOLOv8 state-of-the-art detection |
| **Fast Performance** | WebSocket streaming, ONNX optimization |
| **Production Ready** | Scalable, error handling, logging |
| **Developer Friendly** | Clear documentation, simple API |

---

## 📝 License

[Add your license information here]

---

## 🤝 Contributing

[Add contribution guidelines here]

---

**Built with ❤️ for production-ready object detection**
