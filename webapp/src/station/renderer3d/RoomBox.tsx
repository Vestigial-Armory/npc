import { useMemo } from "react";
import * as THREE from "three";
import type { StationModule } from "../types";
import { MODULE_COLORS, MODULE_HEIGHT } from "../types";

const FLOOR_GAP = 10;

type Props = {
  module: StationModule;
  selected: boolean;
  onClick: () => void;
};

export function RoomBox({ module, selected, onClick }: Props) {
  const { rect, type, damaged, floor } = module;
  const baseColor = MODULE_COLORS[type];
  const h = MODULE_HEIGHT[type] * (damaged ? 0.55 : 1);
  const yBase = floor * FLOOR_GAP;

  const cx = rect.x + rect.w / 2;
  const cz = rect.y + rect.h / 2;

  const material = useMemo(() => {
    const col = damaged
      ? new THREE.Color(baseColor).multiplyScalar(0.35).lerp(new THREE.Color("#4a1010"), 0.5)
      : new THREE.Color(baseColor);
    return new THREE.MeshStandardMaterial({
      color: col,
      transparent: true,
      opacity: damaged ? 0.5 : selected ? 0.95 : 0.78,
      roughness: damaged ? 0.95 : 0.65,
      metalness: damaged ? 0.05 : 0.35,
      emissive: selected ? new THREE.Color(baseColor) : new THREE.Color(0),
      emissiveIntensity: selected ? 0.35 : 0,
      wireframe: damaged,
    });
  }, [baseColor, selected, damaged]);

  return (
    <mesh
      position={[cx, yBase + h / 2, cz]}
      onClick={e => { e.stopPropagation(); onClick(); }}
    >
      <boxGeometry args={[rect.w, h, rect.h]} />
      <primitive object={material} attach="material" />
    </mesh>
  );
}

export { FLOOR_GAP };
