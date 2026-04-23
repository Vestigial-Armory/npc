import type { Corridor, StationModule } from "../types";

type Props = {
  corridor: Corridor;
  modules: StationModule[];
};

function center(m: StationModule) {
  return { x: m.rect.x + m.rect.w / 2, z: m.rect.y + m.rect.h / 2 };
}

export function CorridorTube({ corridor, modules }: Props) {
  const from = modules.find(m => m.id === corridor.fromId);
  const to = modules.find(m => m.id === corridor.toId);
  if (!from || !to) return null;

  const cf = center(from);
  const ct = center(to);

  const dx = ct.x - cf.x;
  const dz = ct.z - cf.z;
  const length = Math.sqrt(dx * dx + dz * dz);
  if (length < 0.1) return null;

  const midX = (cf.x + ct.x) / 2;
  const midZ = (cf.z + ct.z) / 2;
  const angle = Math.atan2(dx, dz);

  return (
    <mesh position={[midX, 0.5, midZ]} rotation={[0, angle, 0]}>
      <boxGeometry args={[corridor.width * 1.2, 1, length]} />
      <meshStandardMaterial color="#2a3560" transparent opacity={0.6} roughness={0.8} metalness={0.2} />
    </mesh>
  );
}
