# YOLO AI Frontend

Modern React frontend for real-time object detection using YOLOv8.

## Features

- **Image Upload & Detection**: Upload images and get annotated results with bounding boxes
- **Video Streaming**: Upload video files and stream with real-time detection
- **IP Camera Streaming**: Connect to IP cameras or local devices for live detection

## Tech Stack

- React 18 + TypeScript
- Vite
- shadcn/ui + Tailwind CSS
- Zustand for state management
- WebRTC for real-time streaming

## Installation

```bash
npm install
```

## Development

```bash
npm run dev
```

The app will be available at `http://localhost:8081`

## Build

```bash
npm run build
```

## Environment Variables

Create a `.env` file:

```env
VITE_API_BASE_URL=http://localhost:8005
VITE_STUN_SERVER=stun:stun.l.google.com:19302
```

## Backend Requirements

Make sure the backend API server is running at `http://localhost:8005` before using the frontend.

