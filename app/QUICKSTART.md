# Quick Start Guide

## Prerequisites

1. **Node.js** (v18 or higher)
   ```bash
   node --version
   ```

2. **npm** (comes with Node.js)
   ```bash
   npm --version
   ```

## Step-by-Step Setup

### 1. Navigate to the app directory
```bash
cd /home/chaos/Documents/chaos/production/yolo-ai/app
```

### 2. Install dependencies
```bash
npm install
```

This will install all required packages:
- React 18
- TypeScript
- Vite
- shadcn/ui dependencies
- WebRTC libraries
- And more...

### 3. Ensure backend is running

**Terminal 1: Start BentoML service**
```bash
cd /home/chaos/Documents/chaos/production/yolo-ai
./scripts/serve_bentoml.sh 1 yolov8-service:latest 3000
```

**Terminal 2: Start API server**
```bash
cd /home/chaos/Documents/chaos/production/yolo-ai
python -m src.api.v1 --host 0.0.0.0 --port 8005
```

### 4. Start the frontend development server
```bash
cd /home/chaos/Documents/chaos/production/yolo-ai/app
npm run dev
```

The frontend will start at: **http://localhost:8081**

## Access the Application

Open your browser and navigate to:
```
http://localhost:8081
```

## Features Available

1. **Image Upload Tab**: Upload images and get annotated results
2. **Video Streaming Tab**: Upload videos and stream with detection
3. **IP Camera Tab**: Connect to cameras for live streaming

## Troubleshooting

### Port already in use
If port 8081 is busy, Vite will automatically use the next available port (8082, 8083, etc.)

### Backend connection errors
- Check that BentoML is running on port 3000
- Check that API server is running on port 8005
- Verify `.env` file has correct API URL

### Module not found errors
```bash
# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### TypeScript errors
These are normal until dependencies are installed. Run `npm install` first.

## Production Build

To build for production:
```bash
npm run build
```

The built files will be in the `dist/` directory.

## Preview Production Build

```bash
npm run preview
```

