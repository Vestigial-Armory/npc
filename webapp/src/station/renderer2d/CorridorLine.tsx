import { Line } from "react-konva";
import type { Corridor, StationModule } from "../types";
import { SCALE } from "./RoomShape";

type Props = {
  corridor: Corridor;
  modules: StationModule[];
};

function center(m: StationModule) {
  return {
    x: (m.rect.x + m.rect.w / 2) * SCALE,
    y: (m.rect.y + m.rect.h / 2) * SCALE,
  };
}

export function CorridorLine({ corridor, modules }: Props) {
  const from = modules.find(m => m.id === corridor.fromId);
  const to = modules.find(m => m.id === corridor.toId);
  if (!from || !to) return null;

  const cf = center(from);
  const ct = center(to);

  return (
    <Line
      points={[cf.x, cf.y, ct.x, ct.y]}
      stroke="#4a5a8a"
      strokeWidth={corridor.width * SCALE * 0.6}
      lineCap="round"
      opacity={0.7}
    />
  );
}
