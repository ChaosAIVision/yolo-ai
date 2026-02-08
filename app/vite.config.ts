import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: '0.0.0.0',  // Allow external connections
    port: 8080,  // Changed to 8080 to match docker port mapping
    proxy: {
      '/api': {
        target: 'http://localhost:5000',  // Backend API port inside container (changed from 8000)
        changeOrigin: true,
        secure: false,
        // Don't rewrite path - backend expects /api/v1/upload
      },
      '/ws': {
        target: 'ws://localhost:5000',  // WebSocket proxy
        ws: true,  // Enable WebSocket proxying
        changeOrigin: true,
        secure: false,
      },
    },
  },
})

