import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    watch: {
      awaitWriteFinish: {
        stabilityThreshold: 250,
        pollInterval: 50
      }
    }
  }
})
