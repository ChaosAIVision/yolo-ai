import { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, Power, PowerOff, Loader2 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { useDetectionStore } from '@/store/detectionStore';
import { useWebRTC } from '@/hooks/useWebRTC';
import { api } from '@/lib/api';

export function IPCameraStream() {
  const [cameraId, setCameraId] = useState('0');
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [fps, setFps] = useState(0);
  const [detectionCount, setDetectionCount] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(Date.now());
  const { setError } = useDetectionStore();
  const { createPeerConnection, createOffer, setAnswer, cleanup } = useWebRTC();

  useEffect(() => {
    return () => {
      cleanup(peerConnectionRef.current);
    };
  }, [cleanup]);

  const connectCamera = useCallback(async () => {
    if (!cameraId.trim()) {
      setError('Please enter a camera ID or URL');
      return;
    }

    setIsConnecting(true);
    setError(null);

    try {
      const pc = createPeerConnection();
      peerConnectionRef.current = pc;

      pc.ontrack = (event) => {
        if (videoRef.current && event.streams[0]) {
          videoRef.current.srcObject = event.streams[0];
          setIsConnected(true);
          setIsConnecting(false);

          // Start FPS counter when video starts
          const updateFPS = () => {
            frameCountRef.current++;
            const now = Date.now();
            if (now - lastTimeRef.current >= 1000) {
              setFps(frameCountRef.current);
              frameCountRef.current = 0;
              lastTimeRef.current = now;
            }
            if (isConnected && videoRef.current && !videoRef.current.paused) {
              requestAnimationFrame(updateFPS);
            }
          };

          if (videoRef.current) {
            videoRef.current.onplay = () => {
              updateFPS();
            };
          }
        }
      };

      const offer = await createOffer(pc);
      if (!offer) {
        throw new Error('Failed to create WebRTC offer');
      }

      const answer = await api.createCameraStreamOffer(
        offer.sdp!,
        offer.type,
        cameraId.trim()
      );

      await setAnswer(pc, answer);

      pc.onconnectionstatechange = () => {
        if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
          disconnectCamera();
          setError('Connection lost. Please try reconnecting.');
        }
      };
    } catch (error) {
      console.error('Error connecting to camera:', error);
      setError(error instanceof Error ? error.message : 'Failed to connect to camera');
      setIsConnecting(false);
      setIsConnected(false);
    }
  }, [cameraId, createOffer, setAnswer, setError, isConnected]);

  const disconnectCamera = useCallback(() => {
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    cleanup(peerConnectionRef.current);
    peerConnectionRef.current = null;
    setIsConnected(false);
    setFps(0);
    setDetectionCount(0);
  }, [cleanup]);

  const handleRetry = () => {
    disconnectCamera();
    setTimeout(() => {
      connectCamera();
    }, 500);
  };

  return (
    <div className="w-full space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>IP Camera Live Streaming</CardTitle>
          <CardDescription>
            Connect to an IP camera or local camera device for real-time object detection.
            Enter camera ID (0, 1, 2...) or RTSP URL.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <Input
                type="text"
                placeholder="Camera ID (e.g., 0) or RTSP URL"
                value={cameraId}
                onChange={(e) => setCameraId(e.target.value)}
                disabled={isConnected || isConnecting}
                className="w-full"
              />
            </div>
            <div className="flex items-center gap-2">
              <Badge
                variant={isConnected ? 'success' : 'secondary'}
                className="gap-2"
              >
                <div
                  className={`h-2 w-2 rounded-full ${
                    isConnected ? 'bg-green-500' : 'bg-gray-400'
                  }`}
                />
                {isConnected ? 'Connected' : 'Disconnected'}
              </Badge>
            </div>
          </div>

          {isConnecting && (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
              <span className="ml-2 text-muted-foreground">Connecting to camera...</span>
            </div>
          )}

          <div className="relative border rounded-lg overflow-hidden bg-black min-h-[400px] flex items-center justify-center">
            {!isConnected && !isConnecting && (
              <div className="text-center text-muted-foreground">
                <Camera className="h-16 w-16 mx-auto mb-4 opacity-50" />
                <p>No camera connected</p>
              </div>
            )}
            <video
              ref={videoRef}
              className="w-full h-auto"
              autoPlay
              playsInline
              muted
            />
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full pointer-events-none"
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="text-sm">
                <span className="font-medium">FPS: </span>
                <span className="text-muted-foreground">{fps}</span>
              </div>
              <div className="text-sm">
                <span className="font-medium">Detections: </span>
                <span className="text-muted-foreground">{detectionCount}</span>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {!isConnected ? (
                <Button
                  onClick={connectCamera}
                  disabled={isConnecting || !cameraId.trim()}
                  className="gap-2"
                >
                  <Power className="h-4 w-4" />
                  Connect
                </Button>
              ) : (
                <>
                  <Button
                    variant="outline"
                    onClick={handleRetry}
                    className="gap-2"
                  >
                    Retry
                  </Button>
                  <Button
                    variant="destructive"
                    onClick={disconnectCamera}
                    className="gap-2"
                  >
                    <PowerOff className="h-4 w-4" />
                    Disconnect
                  </Button>
                </>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

