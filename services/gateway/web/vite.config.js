import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev (`npm run dev`), API calls go to the gateway on :8080.
// In production the gateway serves this built app itself, so no proxy is needed.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/v1": "http://localhost:8080", "/healthz": "http://localhost:8080" },
  },
});
