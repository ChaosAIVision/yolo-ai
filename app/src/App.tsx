import { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { ImageUpload } from './components/ImageUpload';
import { VideoStreamUpload } from './components/VideoStreamUpload';
import { IPCameraStream } from './components/IPCameraStream';
import { useDetectionStore } from './store/detectionStore';

function App() {
  const [activeTab, setActiveTab] = useState('image');
  const { error, setError } = useDetectionStore();

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">YOLO AI</h1>
              <p className="text-sm text-muted-foreground">
                Real-time Object Detection with YOLOv8
              </p>
            </div>
            <div className="text-sm text-muted-foreground">
              API: {import.meta.env.VITE_API_BASE_URL || 'http://localhost:8005'}
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {error && (
          <div className="mb-6 p-4 bg-destructive/10 border border-destructive rounded-lg">
            <p className="text-sm text-destructive">{error}</p>
            <button
              onClick={() => setError(null)}
              className="mt-2 text-xs text-destructive underline"
            >
              Dismiss
            </button>
          </div>
        )}

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="image">Image Upload</TabsTrigger>
            <TabsTrigger value="video">Video Streaming</TabsTrigger>
            <TabsTrigger value="camera">IP Camera</TabsTrigger>
          </TabsList>

          <TabsContent value="image" className="mt-6">
            <ImageUpload />
          </TabsContent>

          <TabsContent value="video" className="mt-6">
            <VideoStreamUpload />
          </TabsContent>

          <TabsContent value="camera" className="mt-6">
            <IPCameraStream />
          </TabsContent>
        </Tabs>
      </main>

      <footer className="border-t mt-12">
        <div className="container mx-auto px-4 py-4">
          <p className="text-sm text-center text-muted-foreground">
            YOLO AI - Production-ready object detection system
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;

