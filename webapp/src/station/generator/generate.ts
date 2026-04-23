import type { StationLayout, StationModule, GenerationParams } from "../types";
import { makeRng } from "./rng";
import { buildBSP } from "./bsp";
import { weightedPickType, MODULE_LABELS } from "./archetypes";
import { buildCorridors } from "./graph";
import { generateRingLayout, generateCircularLayout } from "./layouts";

const SIZE_BOUNDS: Record<GenerationParams["size"], { w: number; h: number }> = {
  small:  { w: 40, h: 40 },
  medium: { w: 70, h: 70 },
  large:  { w: 100, h: 100 },
  huge:   { w: 140, h: 140 },
};

const SIZE_MODULE_COUNT: Record<GenerationParams["size"], [number, number]> = {
  small:  [4, 8],
  medium: [8, 16],
  large:  [14, 24],
  huge:   [20, 36],
};

const MAX_FLOORS: Record<GenerationParams["size"], number> = {
  small: 2,
  medium: 3,
  large: 4,
  huge: 4,
};

function applyBilateralSymmetry(modules: StationModule[], bounds: { w: number; h: number }): StationModule[] {
  const originals = [...modules];
  const mirrored: StationModule[] = originals.map(m => ({
    ...m,
    id: `${m.id}_mir`,
    rect: { ...m.rect, x: bounds.w - m.rect.x - m.rect.w },
  }));
  return [...originals, ...mirrored];
}

function applyRadialSymmetry(modules: StationModule[], bounds: { w: number; h: number }): StationModule[] {
  const cx = bounds.w / 2;
  const cy = bounds.h / 2;
  const result: StationModule[] = [...modules];
  for (const m of modules) {
    for (let rot = 1; rot < 4; rot++) {
      const angle = (Math.PI / 2) * rot;
      const cos   = Math.round(Math.cos(angle));
      const sin   = Math.round(Math.sin(angle));
      const rx    = m.rect.x + m.rect.w / 2 - cx;
      const ry    = m.rect.y + m.rect.h / 2 - cy;
      const nx    = cos * rx - sin * ry + cx - m.rect.w / 2;
      const ny    = sin * rx + cos * ry + cy - m.rect.h / 2;
      result.push({
        ...m,
        id: `${m.id}_r${rot}`,
        rect: { ...m.rect, x: Math.round(nx), y: Math.round(ny) },
      });
    }
  }
  return result;
}

export function generate(params: GenerationParams): StationLayout {
  const rng        = makeRng(params.seed);
  const bounds     = SIZE_BOUNDS[params.size];
  const shape      = params.stationShape ?? "box";
  const floorCount = Math.max(1, Math.min(MAX_FLOORS[params.size], params.floorCount));

  const [minMod, maxMod] = SIZE_MODULE_COUNT[params.size];
  const targetCount = Math.max(minMod, Math.min(maxMod, params.moduleCount));

  // ── Place modules ──────────────────────────────────────────────────
  let modules: StationModule[];

  if (shape === "ring") {
    modules = generateRingLayout(bounds, targetCount, floorCount, params.purpose, rng);
  } else if (shape === "cylinder" || shape === "sphere") {
    modules = generateCircularLayout(bounds, targetCount, floorCount, params.purpose, rng);
  } else {
    // Box: existing BSP approach
    const rects   = buildBSP({ x: 0, y: 0, ...bounds }, targetCount, rng);
    const picked  = rects.slice(0, targetCount);
    modules = picked.map((rect, i) => {
      const type = weightedPickType(params.purpose, () => rng.float());
      return {
        id: `m${i}`,
        type,
        label: rng.pick(MODULE_LABELS[type]),
        rect,
        floor: i % floorCount,
        meta: {},
      };
    });

    // Symmetry only makes sense for box layout
    if (params.symmetry === "bilateral") {
      modules = applyBilateralSymmetry(modules, bounds);
    } else if (params.symmetry === "radial") {
      modules = applyRadialSymmetry(modules, bounds);
    }

    // Clamp mirrored modules to bounds
    modules = modules.map(m => ({
      ...m,
      rect: {
        x: Math.max(0, Math.min(bounds.w - m.rect.w, m.rect.x)),
        y: Math.max(0, Math.min(bounds.h - m.rect.h, m.rect.y)),
        w: m.rect.w,
        h: m.rect.h,
      },
    }));

    // Mark hull-adjacent modules for box stations
    const hullMargin = 3;
    modules = modules.map(m => ({
      ...m,
      hullAirlock: m.type === "airlock" || (
        m.rect.x <= hullMargin ||
        m.rect.y <= hullMargin ||
        m.rect.x + m.rect.w >= bounds.w - hullMargin ||
        m.rect.y + m.rect.h >= bounds.h - hullMargin
      ),
    }));
  }

  // ── Derelict damage ────────────────────────────────────────────────
  if (params.purpose === "derelict") {
    modules = modules.map(m => ({ ...m, damaged: rng.bool(0.4) }));
  }

  // ── Corridors: built per floor so each deck is connected ──────────
  const corridors = [];
  for (let f = 0; f < floorCount; f++) {
    const fm = modules.filter(m => m.floor === f);
    if (fm.length > 1) {
      corridors.push(...buildCorridors(
        fm.map(m => m.id),
        fm.map(m => m.rect),
        params.density,
        rng
      ));
    }
  }

  return {
    seed: params.seed,
    params: { ...params, floorCount, stationShape: shape },
    modules,
    corridors,
    bounds,
  };
}
