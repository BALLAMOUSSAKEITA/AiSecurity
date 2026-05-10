import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const securityHeaders = {
  'Cross-Origin-Opener-Policy':   'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
}

export default defineConfig({
  plugins: [react()],
  preview: {
    host: true,
    allowedHosts: true,
    headers: securityHeaders,
  },
  server: {
    host: true,
    port: 5173,
    headers: securityHeaders,
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
      '/ws':  { target: 'ws://localhost:8000',  ws: true },
    },
  },
})
