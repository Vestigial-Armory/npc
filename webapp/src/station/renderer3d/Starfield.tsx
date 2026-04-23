import { useMemo } from "react";
import * as THREE from "three";

type Props = {
  count?: number;
  radius?: number;
};

export function Starfield({ count = 4000, radius = 900 }: Props) {
  const [positions, sizes] = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const sz = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = radius * (0.85 + Math.random() * 0.15);
      pos[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = r * Math.cos(phi);
      sz[i] = Math.random() < 0.05 ? 3 : Math.random() < 0.2 ? 2 : 1;
    }
    return [pos, sz];
  }, [count, radius]);

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("size", new THREE.BufferAttribute(sizes, 1));
    return geo;
  }, [positions, sizes]);

  return (
    <points geometry={geometry}>
      <pointsMaterial
        color="#e8eeff"
        size={1.8}
        sizeAttenuation={false}
        transparent
        opacity={0.9}
      />
    </points>
  );
}
