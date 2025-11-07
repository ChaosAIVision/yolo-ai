import { create } from 'zustand';

interface DetectionState {
  isProcessing: boolean;
  error: string | null;
  setProcessing: (processing: boolean) => void;
  setError: (error: string | null) => void;
}

export const useDetectionStore = create<DetectionState>((set) => ({
  isProcessing: false,
  error: null,
  setProcessing: (processing) => set({ isProcessing: processing }),
  setError: (error) => set({ error }),
}));

