import path from 'path';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

// Standalone build config. This app was exported from a Replit pnpm monorepo;
// it has since been de-monorepo'd so `npm install && npm run build` works on
// its own (see README). Two changes vs the original export:
//   * PORT / BASE_PATH are optional — they default to sane local/production
//     values instead of throwing, so a bare `npm run build` succeeds.
//   * The Replit-only vite plugins (cartographer, dev-banner,
//     runtime-error-modal) were dropped; they only ran inside Replit and their
//     versions came from the (now-absent) workspace catalog.
// The dev and preview servers bind to localhost with Vite's default Host-header
// check. Replit needed 0.0.0.0 + `allowedHosts: true`; off Replit that exposes
// the dev server, and every file-access advisory Vite ships against it, to the
// LAN and to DNS-rebinding pages. Pass `--host` to share it on a trusted network.
const port = Number(process.env.PORT ?? '5173');
const basePath = process.env.BASE_PATH ?? '/';

export default defineConfig({
  base: basePath,
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, 'src'),
      '@assets': path.resolve(import.meta.dirname, 'attached_assets'),
    },
    dedupe: ['react', 'react-dom'],
  },
  root: path.resolve(import.meta.dirname),
  build: {
    outDir: path.resolve(import.meta.dirname, 'dist/public'),
    emptyOutDir: true,
  },
  server: {
    port,
    strictPort: true,
    fs: {
      strict: true,
    },
  },
  preview: {
    port,
  },
});
