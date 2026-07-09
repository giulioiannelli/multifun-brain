import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev: the Vite server proxies /api to the FastAPI backend on :8000.
// Build: emits to dist/, which FastAPI serves as static files in local/prod.
// Vendor chunks are split so the browser caches Plotly/Cytoscape separately and
// parses them in parallel; Cytoscape is also lazy-loaded (only on the Network tab).
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
    chunkSizeWarningLimit: 2000,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules")) {
            if (id.includes("plotly.js")) return "plotly";
            if (id.includes("cytoscape")) return "cytoscape";
            if (id.includes("/react") || id.includes("/react-dom")) return "react";
          }
        },
      },
    },
  },
});
