import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Download, Loader2 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { useDetectionStore } from '@/store/detectionStore';
import { api } from '@/lib/api';

export function ImageUpload() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { setError } = useDetectionStore();

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please upload a valid image file (JPEG, PNG)');
      return;
    }

    // Preview uploaded image
    const reader = new FileReader();
    reader.onload = () => {
      setUploadedImage(reader.result as string);
    };
    reader.readAsDataURL(file);

    // Upload and get annotated image
    setIsLoading(true);
    setError(null);

    try {
      const blob = await api.uploadImage(file);
      const imageUrl = URL.createObjectURL(blob);
      setAnnotatedImage(imageUrl);
    } catch (error) {
      console.error('Error uploading image:', error);
      setError(error instanceof Error ? error.message : 'Failed to upload image');
      setAnnotatedImage(null);
    } finally {
      setIsLoading(false);
    }
  }, [setError]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png'],
    },
    maxFiles: 1,
  });

  const handleDownload = () => {
    if (annotatedImage) {
      const link = document.createElement('a');
      link.href = annotatedImage;
      link.download = 'annotated-image.jpg';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <div className="w-full space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Image Upload & Detection</CardTitle>
          <CardDescription>
            Upload an image to detect objects with YOLOv8. Supports JPEG and PNG formats.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div
            {...getRootProps()}
            className={`
              border-2 border-dashed rounded-lg p-12 text-center cursor-pointer
              transition-colors
              ${isDragActive ? 'border-primary bg-primary/5' : 'border-muted-foreground/25'}
              hover:border-primary hover:bg-primary/5
            `}
          >
            <input {...getInputProps()} />
            <Upload className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
            {isDragActive ? (
              <p className="text-lg font-medium">Drop the image here...</p>
            ) : (
              <div>
                <p className="text-lg font-medium mb-2">
                  Drag & drop an image here, or click to select
                </p>
                <p className="text-sm text-muted-foreground">
                  Supports JPEG, PNG formats
                </p>
              </div>
            )}
          </div>

          {isLoading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
              <span className="ml-2 text-muted-foreground">Processing image...</span>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {uploadedImage && (
              <div className="space-y-2">
                <h3 className="text-sm font-medium">Original Image</h3>
                <div className="border rounded-lg overflow-hidden">
                  <img
                    src={uploadedImage}
                    alt="Uploaded"
                    className="w-full h-auto object-contain"
                  />
                </div>
              </div>
            )}

            {annotatedImage && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-medium">Annotated Result</h3>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleDownload}
                    className="gap-2"
                  >
                    <Download className="h-4 w-4" />
                    Download
                  </Button>
                </div>
                <div className="border rounded-lg overflow-hidden">
                  <img
                    src={annotatedImage}
                    alt="Annotated"
                    className="w-full h-auto object-contain"
                  />
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

