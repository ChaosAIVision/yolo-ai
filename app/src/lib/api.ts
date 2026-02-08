// Use proxy in dev mode, or full URL in production
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || (import.meta.env.DEV ? '' : 'http://localhost:5000');

/**
 * Upload an image for object detection
 * @param file - Image file to upload
 * @returns Promise resolving to a Blob containing the annotated image
 */
export async function uploadImage(file: File): Promise<Blob> {
  const formData = new FormData();
  formData.append('image', file);

  const response = await fetch(`${API_BASE_URL}/api/v1/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to upload image: ${response.status} ${errorText}`);
  }

  return await response.blob();
}

export const api = {
  uploadImage,
};

