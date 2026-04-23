import type { Rng } from "./rng";

export type BSPRect = { x: number; y: number; w: number; h: number };

type BSPNode =
  | { kind: "leaf"; rect: BSPRect }
  | { kind: "split"; rect: BSPRect; left: BSPNode; right: BSPNode; horizontal: boolean; splitAt: number };

const MIN_SIZE = 4;

function split(rect: BSPRect, rng: Rng, depth: number, maxDepth: number): BSPNode {
  const tooSmall = rect.w < MIN_SIZE * 2 + 1 && rect.h < MIN_SIZE * 2 + 1;
  if (depth >= maxDepth || tooSmall) {
    return { kind: "leaf", rect };
  }

  const canH = rect.h >= MIN_SIZE * 2 + 1;
  const canV = rect.w >= MIN_SIZE * 2 + 1;
  const horizontal = canH && canV ? rng.bool() : canH;

  if (!canH && !canV) return { kind: "leaf", rect };

  if (horizontal) {
    const minCut = rect.y + MIN_SIZE;
    const maxCut = rect.y + rect.h - MIN_SIZE;
    if (minCut >= maxCut) return { kind: "leaf", rect };
    const splitAt = rng.int(minCut, maxCut);
    return {
      kind: "split", rect, horizontal: true, splitAt,
      left:  split({ x: rect.x, y: rect.y, w: rect.w, h: splitAt - rect.y }, rng, depth + 1, maxDepth),
      right: split({ x: rect.x, y: splitAt, w: rect.w, h: rect.h - (splitAt - rect.y) }, rng, depth + 1, maxDepth),
    };
  } else {
    const minCut = rect.x + MIN_SIZE;
    const maxCut = rect.x + rect.w - MIN_SIZE;
    if (minCut >= maxCut) return { kind: "leaf", rect };
    const splitAt = rng.int(minCut, maxCut);
    return {
      kind: "split", rect, horizontal: false, splitAt,
      left:  split({ x: rect.x, y: rect.y, w: splitAt - rect.x, h: rect.h }, rng, depth + 1, maxDepth),
      right: split({ x: splitAt, y: rect.y, w: rect.w - (splitAt - rect.x), h: rect.h }, rng, depth + 1, maxDepth),
    };
  }
}

function collectLeaves(node: BSPNode): BSPRect[] {
  if (node.kind === "leaf") return [node.rect];
  return [...collectLeaves(node.left), ...collectLeaves(node.right)];
}

function shrinkRect(rect: BSPRect, rng: Rng): BSPRect {
  const margin = 1;
  const maxShrink = 2;
  const shrinkW = rng.int(0, maxShrink);
  const shrinkH = rng.int(0, maxShrink);
  return {
    x: rect.x + margin + Math.floor(shrinkW / 2),
    y: rect.y + margin + Math.floor(shrinkH / 2),
    w: Math.max(MIN_SIZE - 1, rect.w - margin * 2 - shrinkW),
    h: Math.max(MIN_SIZE - 1, rect.h - margin * 2 - shrinkH),
  };
}

export function buildBSP(bounds: BSPRect, moduleCount: number, rng: Rng): BSPRect[] {
  const depth = Math.ceil(Math.log2(moduleCount + 1));
  const tree = split(bounds, rng, 0, depth);
  const leaves = collectLeaves(tree);
  return leaves.map(r => shrinkRect(r, rng));
}
