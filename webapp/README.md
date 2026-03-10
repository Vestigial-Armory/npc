# NPC Action Generator (Static + Mobile-first)

This app runs fully in-browser and is designed for static hosting (GitHub Pages, SiteGround static deployment, or any CDN).

## Current behavior

- Loads NPC character sheets from CSV using section and field markers.
- No trait rolling in UI.
- Shows only the first 5 identifying traits as a preview.
- Supports prompt tags:
  - `#Name#` for connection references
  - `*item*` for inventory references
- Generates a narrative response plus a list of effects.
- Applies effects back into the sheet and logs an event in `History` with importance and effect summary.
- Uses WebLLM when compatible.
- Uses iOS-lite local model path when iOS low-GPU limits block WebLLM.
- Falls back to deterministic heuristic output if model generation fails.

## Character sheet CSV format

- `!!Section Name` starts a section.
- `!Field` cells define the field header row for that section.
- All following rows are data until the next `!!Section`.

Example template is included at:

- `public/npc_char_sheet.csv`

## Development

- Install: `npm install`
- Dev server: `npm run dev`
- Lint: `npm run lint`
- Build: `npm run build`

## Deployment notes

- Deploy the generated `dist/` folder to any static host.
- For project-subpath hosting (like `https://<user>.github.io/<repo>/`), set `VITE_BASE_PATH` during build.

## GitHub Pages setup

1. In repo settings open `Settings -> Pages`.
2. Set `Source` to `GitHub Actions`.
3. Wait for workflow `Deploy webapp to GitHub Pages` to finish.
4. Open: `https://vestigial-armory.github.io/npc/`

## Mobile testing

### iPad

- Open app URL in Safari.
- If WebLLM is incompatible due to low GPU limits, app will show iOS-lite mode and still generate actions locally.

### Android

- Existing WebLLM path remains unchanged.
- If device/browser supports required WebGPU limits, standard model flow works.

## Troubleshooting

- Error like `requested maxComputeWorkgroupStorageSize exceeds limit` means WebLLM cannot run on that device/browser limit.
- On iOS low-GPU devices, app switches to iOS-lite local model path automatically.
- If local model load fails, generation still falls back to deterministic narrative + effects output.
