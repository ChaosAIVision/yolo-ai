import { useState, useRef, useCallback } from 'react';
import { Camera, Power, PowerOff, Loader2 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { useDetectionStore } from '@/store/detectionStore';

export function IPCameraStream() {
  const [cameraId, setCameraId] = useState('0');
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [fps, setFps] = useState(0);
  const [detectionCount, setDetectionCount] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { setError } = useDetectionStore();

  const connectCamera = useCallback(async () => {
    if (!cameraId.trim()) {
      setError('Please enter a camera ID or URL');
      return;
    }

    // Feature not implemented - UI only
    setError('Camera streaming feature is not yet implemented. This is a UI placeholder.');
    setIsConnecting(false);
    setIsConnected(false);
    
    // Optional: You can remove the error and just show a message
    // Or keep the UI in a "not implemented" state
  }, [cameraId, setError]);

  const disconnectCamera = useCallback(() => {
    // UI only - just reset state
    setIsConnected(false);
    setIsConnecting(false);
    setFps(0);
    setDetectionCount(0);
    setError(null);
    
    // Clear canvas if exists
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
  }, [setError]);

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
            Enter camera ID (0, 1, 2...) or RTSP URL. (UI Only - Feature coming soon)
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
            <canvas
              ref={canvasRef}
              style={{ 
                maxWidth: '100%',
                height: 'auto',
                display: 'block',
                imageRendering: 'auto'
              }}
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

