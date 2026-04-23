import { useMemo } from "react";
import * as THREE from "three";
import type { StationModule } from "../types";
import { MODULE_COLORS, MODULE_HEIGHT } from "../types";

type Props = {
  module: StationModule;
  selected: boolean;
  onClick: () => void;
};

export function RoomBox({ module, selected, onClick }: Props) {
  const { rect, type, damaged } = module;
  const baseColor = MODULE_COLORS[type];
  const h = MODULE_HEIGHT[type] * (damaged ? 0.55 : 1);

  const cx = rect.x + rect.w / 2;
  const cz = rect.y + rect.h / 2;

  const material = useMemo(() => {
    const col = damaged
      ? new THREE.Color(baseColor).multiplyScalar(0.35).lerp(new THREE.Color("#4a1010"), 0.5)
      : new THREE.Color(baseColor);

    return new THREE.MeshStandardMaterial({
      color: col,
      transparent: true,
      opacity: damaged ? 0.5 : selected ? 0.95 : 0.75,
      roughness: damaged ? 0.95 : 0.7,
      metalness: damaged ? 0.05 : 0.3,
      emissive: selected ? new THREE.Color(baseColor) : new THREE.Color(0),
      emissiveIntensity: selected ? 0.3 : 0,
      wireframe: damaged,
    });
  }, [baseColor, selected, damaged]);

  return (
    <mesh
      position={[cx, h / 2, cz]}
      onClick={e => { e.stopPropagation(); onClick(); }}
    >
      <boxGeometry args={[rect.w, h, rect.h]} />
      <primitive object={material} attach="material" />
    </mesh>
  );
}
