import type { BSPRect } from "./bsp";
import type { Rng } from "./rng";
import type { Corridor, Vec2 } from "../types";

function rectCenter(r: BSPRect): Vec2 {
  return { x: r.x + r.w / 2, y: r.y + r.h / 2 };
}

function rectsAreAdjacent(a: BSPRect, b: BSPRect): boolean {
  const gap = 3;
  const hOverlap = a.x < b.x + b.w + gap && b.x < a.x + a.w + gap;
  const vOverlap = a.y < b.y + b.h + gap && b.y < a.y + a.h + gap;
  return hOverlap && vOverlap;
}

function dist(a: Vec2, b: Vec2): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// Prim's MST on the adjacency graph
function buildMST(rects: BSPRect[]): [number, number][] {
  const n = rects.length;
  if (n <= 1) return [];
  const centers = rects.map(rectCenter);
  const inTree = new Set<number>([0]);
  const edges: [number, number][] = [];

  while (inTree.size < n) {
    let best: [number, number] | null = null;
    let bestD = Infinity;
    for (const i of inTree) {
      for (let j = 0; j < n; j++) {
        if (inTree.has(j)) continue;
        const d = dist(centers[i], centers[j]);
        if (d < bestD) { bestD = d; best = [i, j]; }
      }
    }
    if (!best) break;
    inTree.add(best[1]);
    edges.push(best);
  }
  return edges;
}

export function buildCorridors(
  ids: string[],
  rects: BSPRect[],
  density: number,
  rng: Rng
): Corridor[] {
  const mstEdges = buildMST(rects);
  const edgeSet = new Set(mstEdges.map(([a, b]) => `${Math.min(a,b)}-${Math.max(a,b)}`));

  // Add extra edges for loops based on density
  const extraCount = Math.round(density * rects.length);
  const candidates: [number, number][] = [];
  for (let i = 0; i < rects.length; i++) {
    for (let j = i + 1; j < rects.length; j++) {
      const key = `${i}-${j}`;
      if (!edgeSet.has(key) && rectsAreAdjacent(rects[i], rects[j])) {
        candidates.push([i, j]);
      }
    }
  }
  const shuffled = rng.shuffle(candidates);
  for (let k = 0; k < Math.min(extraCount, shuffled.length); k++) {
    const [a, b] = shuffled[k];
    const key = `${Math.min(a,b)}-${Math.max(a,b)}`;
    if (!edgeSet.has(key)) {
      edgeSet.add(key);
      mstEdges.push([a, b]);
    }
  }

  return mstEdges.map(([a, b], i): Corridor => ({
    id: `c${i}`,
    fromId: ids[a],
    toId: ids[b],
    width: 1,
    waypoints: [],
  }));
}
