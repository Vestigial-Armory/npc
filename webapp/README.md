# NPC Action Generator (Static + Mobile-first)

This app runs fully in the browser and is designed for static hosting (GitHub Pages, SiteGround static deployment, or any CDN).

It supports:

- CSV upload for NPC trait tables
- Weighted trait rolls
- Situation prompt input
- In-browser LLM action generation using WebLLM + WebGPU
- Automatic rules-only fallback when WebGPU/model loading is unavailable

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
