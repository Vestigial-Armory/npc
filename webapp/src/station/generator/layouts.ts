import type { Rng } from "./rng";
import type { StationModule, GenerationParams } from "../types";
import { weightedPickType, MODULE_LABELS } from "./archetypes";

type Bounds = { w: number; h: number };

// ── Ring layout ────────────────────────────────────────────────────────
// Modules arranged in an annulus. Hub command centre in the middle.
// Adjacent ring modules are geometrically close so the MST corridor
// builder naturally produces ring-shaped corridors.

export function generateRingLayout(
  bounds: Bounds,
  moduleCount: number,
  floorCount: number,
  purpose: GenerationParams["purpose"],
  rng: Rng
): StationModule[] {
  const cx = bounds.w / 2;
  const cy = bounds.h / 2;
  const maxR   = Math.min(cx, cy) * 0.82;
  const ringW  = maxR * 0.30;
  const midR   = maxR - ringW / 2;

  // Tangential arc-length per module → module width; radial depth → height
  const arcLen = (2 * Math.PI * midR) / moduleCount;
  const modW   = Math.max(4, Math.min(Math.round(arcLen * 0.80), 12));
  const modH   = Math.max(4, Math.min(Math.round(ringW * 0.76), 8));

  const modules: StationModule[] = [];

  for (let i = 0; i < moduleCount; i++) {
    const angle = (i / moduleCount) * Math.PI * 2 - Math.PI / 2;
    const type  = weightedPickType(purpose, () => rng.float());
    const mx    = cx + midR * Math.cos(angle) - modW / 2;
    const my    = cy + midR * Math.sin(angle) - modH / 2;
    modules.push({
      id: `m${i}`,
      type,
      label: rng.pick(MODULE_LABELS[type]),
      rect: {
        x: Math.round(mx),
        y: Math.round(my),
        w: modW,
        h: modH,
      },
      floor: i % floorCount,
      hullAirlock: true,   // all ring modules face the hull
      meta: {},
    });
  }

  // Central command hub
  const hubSize = Math.round(ringW * 0.62);
  modules.push({
    id: "mhub",
    type: "command",
    label: "Command Hub",
    rect: {
      x: Math.round(cx - hubSize / 2),
      y: Math.round(cy - hubSize / 2),
      w: hubSize,
      h: hubSize,
    },
    floor: 0,
    hullAirlock: false,
    meta: {},
  });

  return modules;
}

// ── Circular layout ────────────────────────────────────────────────────
// Hub at centre, inner ring, outer ring — used for sphere and cylinder.

export function generateCircularLayout(
  bounds: Bounds,
  moduleCount: number,
  floorCount: number,
  purpose: GenerationParams["purpose"],
  rng: Rng
): StationModule[] {
  const cx   = bounds.w / 2;
  const cy   = bounds.h / 2;
  const maxR = Math.min(cx, cy) * 0.84;

  const modules: StationModule[] = [];
  let id = 0;

  // Hub
  const hubSize = Math.round(maxR * 0.20);
  modules.push({
    id: `m${id++}`,
    type: "command",
    label: "Command Centre",
    rect: {
      x: Math.round(cx - hubSize / 2),
      y: Math.round(cy - hubSize / 2),
      w: hubSize,
      h: hubSize,
    },
    floor: 0,
    hullAirlock: false,
    meta: {},
  });

  const remaining  = moduleCount - 1;
  const innerCount = Math.max(1, Math.round(remaining * 0.40));
  const outerCount = remaining - innerCount;

  function placeRing(
    count: number,
    r: number,
    modW: number,
    modH: number,
    isHull: boolean,
    floorOffset: number,
  ) {
    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2 - Math.PI / 2;
      const type  = weightedPickType(purpose, () => rng.float());
      modules.push({
        id: `m${id++}`,
        type,
        label: rng.pick(MODULE_LABELS[type]),
        rect: {
          x: Math.round(cx + r * Math.cos(angle) - modW / 2),
          y: Math.round(cy + r * Math.sin(angle) - modH / 2),
          w: modW,
          h: modH,
        },
        floor: (id + floorOffset) % floorCount,
        hullAirlock: isHull,
        meta: {},
      });
    }
  }

  if (innerCount > 0) placeRing(innerCount, maxR * 0.44, 6, 6, false, 1);
  if (outerCount > 0) placeRing(outerCount, maxR * 0.80, 7, 5, true,  0);

  return modules;
}
