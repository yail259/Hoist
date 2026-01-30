import { defineConfig } from 'astro/config';
import preact from '@astrojs/preact';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
  integrations: [preact()],
  vite: {
    plugins: [wasm(), topLevelAwait()],
  },
});
