# NPC Action Generator (Static + Mobile-first)

This app runs fully in the browser and is designed for static hosting (GitHub Pages, SiteGround static deployment, or any CDN).

It supports:

- CSV upload for NPC trait tables
- Weighted trait rolls
- Situation prompt input
- In-browser LLM action generation using WebLLM + WebGPU
- Automatic rules-only fallback when WebGPU/model loading is unavailable
- Built-in sample traits loaded at startup (no upload required for quick testing)

## CSV format

Use these columns (case-insensitive):

- `trait` (or `category` / `attribute`)
- `value` (or `option` / `result`)
- `weight` (optional; defaults to `1`)

Example:

trait,value,weight
personality,curious,3
personality,suspicious,1
motivation,protect family,4

## Development

Install dependencies:

`npm install`

Run local dev server:

`npm run dev`

Run lint:

`npm run lint`

Build static output:

`npm run build`

## Deployment notes

- Deploy the generated `dist/` folder to any static host.
- For GitHub Pages under a project subpath, set Vite `base` in `vite.config.ts`.
- WebLLM requires WebGPU support; devices without WebGPU use fallback mode.

## iPad and Android testing instructions

### Quick test (no upload needed)

1. Open your deployed HTTPS URL on iPad Safari or Android Chrome.
2. Confirm the app loads with `Loaded file: built-in sample`.
3. Confirm traits are already rolled.
4. Tap `Generate NPC Action`.
5. Verify `Action output` is populated.

This path works even without model loading because fallback mode is automatic.

### Optional: test WebLLM model loading

1. Tap `Load Model`.
2. Wait for status updates.
3. If load succeeds, generate again and confirm model path output.
4. If load fails, fallback mode remains available and the app should still generate output.

### Optional: upload your own CSV

1. Download and edit `public/sample_traits.csv` format as needed.
2. Upload your CSV in the app.
3. Tap `Roll NPC Traits`.
4. Generate an action to validate your custom table.
