# Frontend Setup Instructions

## Quick Start

1. **Install Dependencies**
   ```bash
   cd app
   npm install
   ```

2. **Configure Environment**
   - The `.env` file is already created with default values
   - Modify if needed:
     ```env
     VITE_API_BASE_URL=http://localhost:8005
     VITE_STUN_SERVER=stun:stun.l.google.com:19302
     ```

3. **Start Development Server**
   ```bash
   npm run dev
   ```
   The app will be available at `http://localhost:8081`

4. **Ensure Backend is Running**
   - Make sure the BentoML service is running on port 3000
   - Make sure the API server is running on port 8005

## Project Structure

```
app/
├── src/
│   ├── components/
│   │   ├── ui/              # shadcn/ui components
│   │   ├── ImageUpload.tsx  # Image upload feature
│   │   ├── VideoStreamUpload.tsx  # Video streaming feature
│   │   └── IPCameraStream.tsx     # IP camera feature
│   ├── hooks/
│   │   └── useWebRTC.ts     # WebRTC hook
│   ├── lib/
│   │   ├── api.ts          # API utilities
│   │   └── utils.ts        # Utility functions
│   ├── store/
│   │   └── detectionStore.ts  # Zustand store
│   ├── App.tsx             # Main app component
│   ├── main.tsx            # Entry point
│   └── index.css           # Global styles
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.js
```

## Features Implemented

### 1. Image Upload & Detection
- Drag-and-drop image upload
- Real-time image annotation with bounding boxes
- Download annotated images
- Support for JPEG and PNG formats

### 2. Video Streaming
- Upload video files (MP4, AVI, MOV)
- Real-time WebRTC streaming with detection
- FPS counter and detection stats
- Play/Pause/Stop controls

### 3. IP Camera Streaming
- Connect to IP cameras or local devices
- Real-time live streaming with detection
- Connection status indicator
- Retry mechanism for failed connections

## Notes

- The video upload feature currently uses a placeholder path. You may need to implement actual file upload to the server first.
- WebRTC requires HTTPS in production (localhost works for development)
- Make sure CORS is properly configured on the backend
- The app uses shadcn/ui components styled with Tailwind CSS

## Troubleshooting

1. **Module not found errors**: Run `npm install` again
2. **API connection errors**: Check that backend is running on port 8005
3. **WebRTC connection fails**: Check STUN server configuration
4. **TypeScript errors**: Make sure all dependencies are installed

