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
  const { rect, type } = module;
  const color = MODULE_COLORS[type];
  const h = MODULE_HEIGHT[type];

  const cx = rect.x + rect.w / 2;
  const cz = rect.y + rect.h / 2;

  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(color),
        transparent: true,
        opacity: selected ? 0.95 : 0.75,
        roughness: 0.7,
        metalness: 0.3,
        emissive: selected ? new THREE.Color(color) : new THREE.Color(0),
        emissiveIntensity: selected ? 0.3 : 0,
      }),
    [color, selected]
  );

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
