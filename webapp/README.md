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
- For project-subpath hosting (like `https://<user>.github.io/<repo>/`), set `VITE_BASE_PATH` during build.
- WebLLM requires WebGPU support; devices without WebGPU use fallback mode.

## GitHub Pages setup (fixes 404 / blank page)

Your app source is in `webapp/`, so Pages cannot serve it directly from repo root without a build/deploy step.

1. Push this repository branch with `.github/workflows/deploy-pages.yml`.
2. In GitHub repo settings, open `Settings -> Pages`.
3. Under Build and deployment, set `Source` to `GitHub Actions`.
4. Let the `Deploy webapp to GitHub Pages` workflow finish.
5. Open: `https://vestigial-armory.github.io/npc/`

Notes:

- `404` at `/npc/` happens when no built `index.html` is deployed.
- Blank page at `/webapp` happens because raw Vite source (`src/main.tsx`) is not static production output.
- This workflow builds `webapp` and deploys `webapp/dist` with `VITE_BASE_PATH` set from your repo name (for this repo, `/npc/`) so asset URLs resolve correctly.

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

Troubleshooting:

- If you see `requested maxComputeWorkgroupStorageSize exceeds limit`, your device GPU limit is below WebLLM runtime requirements.
- The app now pre-checks this and disables model load when incompatible.
- You can still use `Generate NPC Action` through rules-only fallback mode.

### Optional: upload your own CSV

1. Download and edit `public/sample_traits.csv` format as needed.
2. Upload your CSV in the app.
3. Tap `Roll NPC Traits`.
4. Generate an action to validate your custom table.
