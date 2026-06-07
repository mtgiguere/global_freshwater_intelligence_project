import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// VITE_BASE_PATH controls the base URL for the built app.
// - Local dev / custom domain: leave unset (defaults to '/')
// - GitHub Pages: set to '/global_freshwater_intelligence_project/'
//   (the GitHub Actions deploy workflow sets this automatically)
const base = process.env.VITE_BASE_PATH ?? '/'

export default defineConfig({
  base,
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'lcov'],
      exclude: ['src/main.tsx', 'src/test/**'],
    },
  },
})
