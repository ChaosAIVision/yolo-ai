import { useState, useRef, useEffect, useCallback } from 'react';
import { Play, Square, Loader2, Youtube } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { useDetectionStore } from '@/store/detectionStore';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || (import.meta.env.DEV ? '' : 'http://localhost:5000');

export function VideoStreamUpload() {
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [fps, setFps] = useState(0);
  const [detectionCount, setDetectionCount] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(Date.now());
  const animationFrameRef = useRef<number | null>(null);
  const { setError } = useDetectionStore();

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  const validateYouTubeUrl = (url: string): boolean => {
    const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+/;
    return youtubeRegex.test(url);
  };

  const startStreaming = useCallback(async () => {
    if (!youtubeUrl.trim()) {
      setError('Please enter a YouTube URL');
      return;
    }

    if (!validateYouTubeUrl(youtubeUrl)) {
      setError('Please enter a valid YouTube URL (e.g., https://www.youtube.com/watch?v=...)');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Connect to WebSocket
      // In dev mode, use proxy (ws://localhost:8080/ws/youtube)
      // In production, use API_BASE_URL
      let wsUrl: string;
      if (import.meta.env.DEV || !API_BASE_URL) {
        // Dev mode: use proxy via Vite dev server
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        wsUrl = `${protocol}//${window.location.host}/ws/youtube`;
      } else {
        // Production: use API_BASE_URL
        wsUrl = API_BASE_URL.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws/youtube';
      }
      
      console.log('Connecting to WebSocket:', wsUrl);
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected');
        // Send start message with YouTube URL
        ws.send(JSON.stringify({
          type: 'start',
          youtube_url: youtubeUrl.trim()
        }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'ready') {
            console.log(`Stream ready: ${data.width}x${data.height}`);
            // Set canvas size
            if (canvasRef.current) {
              canvasRef.current.width = data.width;
              canvasRef.current.height = data.height;
              // Set display size to match aspect ratio
              const aspectRatio = data.width / data.height;
              const maxWidth = 1280; // Max display width
              const displayWidth = Math.min(maxWidth, data.width);
              const displayHeight = displayWidth / aspectRatio;
              canvasRef.current.style.width = `${displayWidth}px`;
              canvasRef.current.style.height = `${displayHeight}px`;
            }
            setIsStreaming(true);
            setIsLoading(false);
          } else if (data.type === 'frame') {
            // Decode base64 frame and display
            const img = new Image();
            img.onload = () => {
              if (canvasRef.current) {
                const ctx = canvasRef.current.getContext('2d');
                if (ctx) {
                  // Clear canvas before drawing new frame
                  ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                  
                  // Draw image with exact size (no scaling)
                  ctx.drawImage(img, 0, 0, canvasRef.current.width, canvasRef.current.height);
                  
                  // Update FPS counter
                  frameCountRef.current++;
                  const now = Date.now();
                  if (now - lastTimeRef.current >= 1000) {
                    setFps(frameCountRef.current);
                    frameCountRef.current = 0;
                    lastTimeRef.current = now;
                  }
                }
              }
            };
            img.onerror = (error) => {
              console.error('Error loading image:', error);
            };
            img.src = `data:image/jpeg;base64,${data.data}`;
          } else if (data.type === 'error') {
            setError(data.message || 'Stream error');
            setIsLoading(false);
            setIsStreaming(false);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('WebSocket connection error');
        setIsLoading(false);
        setIsStreaming(false);
      };

      ws.onclose = () => {
        console.log('WebSocket closed');
        setIsStreaming(false);
        setIsLoading(false);
      };
    } catch (error) {
      console.error('Error starting stream:', error);
      setError(error instanceof Error ? error.message : 'Failed to start streaming');
      setIsLoading(false);
    }
  }, [youtubeUrl, setError]);

  const stopStreaming = useCallback(async () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        // Send stop message to backend
        wsRef.current.send(JSON.stringify({ type: 'stop' }));
        // Wait a bit to ensure backend receives the message before closing
        await new Promise(resolve => setTimeout(resolve, 100));
      } catch (error) {
        console.error('Error sending stop message:', error);
      }
    }
    
    // Close WebSocket connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    // Clear canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
    
    // Reset state
    setIsStreaming(false);
    setFps(0);
    setDetectionCount(0);
  }, []);

  return (
    <div className="w-full space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>YouTube Stream & Real-time Detection</CardTitle>
          <CardDescription>
            Enter a YouTube URL to stream with real-time object detection. 
            Frame skipping enabled: processes every 3rd frame at 20 FPS.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <Input
                type="text"
                placeholder="Enter YouTube URL (e.g., https://www.youtube.com/watch?v=...)"
                value={youtubeUrl}
                onChange={(e) => setYoutubeUrl(e.target.value)}
                disabled={isStreaming || isLoading}
                className="w-full"
              />
            </div>
            <Youtube className="h-5 w-5 text-muted-foreground" />
          </div>

          {isLoading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
              <span className="ml-2 text-muted-foreground">Connecting to YouTube stream...</span>
            </div>
          )}

          <div className="relative border rounded-lg overflow-hidden bg-black min-h-[400px] flex items-center justify-center">
            {!isStreaming && !isLoading && (
              <div className="text-center text-muted-foreground">
                <Youtube className="h-16 w-16 mx-auto mb-4 opacity-50" />
                <p>No stream connected</p>
                <p className="text-sm mt-2">Enter a YouTube URL and click Start Stream</p>
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
              <div className="text-xs text-muted-foreground">
                Frame skip: Every 3rd frame processed
              </div>
            </div>

            <div className="flex items-center gap-2">
              {!isStreaming ? (
                <Button
                  onClick={startStreaming}
                  disabled={!youtubeUrl.trim() || isLoading}
                  className="gap-2"
                >
                  <Play className="h-4 w-4" />
                  Start Stream
                </Button>
              ) : (
                <Button
                  variant="destructive"
                  onClick={stopStreaming}
                  className="gap-2"
                >
                  <Square className="h-4 w-4" />
                  Stop
                </Button>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
