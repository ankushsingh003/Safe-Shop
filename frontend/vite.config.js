
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3001,
    proxy: {
      '/predict':  'http://localhost:8000',
      '/forecast': 'http://localhost:8000',
      '/health':   'http://localhost:8000',
      '/ask':      'http://localhost:8000',
      '/rag':      'http://localhost:8000',
    }
  }
})