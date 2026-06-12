import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev: the Vite server proxies /api to the FastAPI backend on :8000.
// Build: emits to dist/, which FastAPI serves as static files in local/prod.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 4000,
  },
});
