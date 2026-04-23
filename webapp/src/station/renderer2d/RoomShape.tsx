import { Rect, Text, Group, Line } from "react-konva";
import type { StationModule } from "../types";
import { MODULE_COLORS } from "../types";

const SCALE = 10;

type Props = {
  module: StationModule;
  selected: boolean;
  onClick: () => void;
  onDragEnd: (x: number, y: number) => void;
};

export function RoomShape({ module, selected, onClick, onDragEnd }: Props) {
  const { rect, type, label, damaged, hullAirlock } = module;
  const baseColor = MODULE_COLORS[type];
  const fillColor = damaged ? "#2a1a1a" : baseColor + "44";
  const strokeColor = damaged ? "#a03030" : selected ? "#ffffff" : baseColor;
  const strokeWidth = selected ? 2.5 : damaged ? 2 : 1.5;
  const w = rect.w * SCALE;
  const h = rect.h * SCALE;

  return (
    <Group
      x={rect.x * SCALE}
      y={rect.y * SCALE}
      draggable
      onClick={onClick}
      onTap={onClick}
      onDragEnd={e => {
        const newX = Math.round(e.target.x() / SCALE);
        const newY = Math.round(e.target.y() / SCALE);
        onDragEnd(newX, newY);
        e.target.x(newX * SCALE);
        e.target.y(newY * SCALE);
      }}
    >
      <Rect
        width={w}
        height={h}
        fill={fillColor}
        stroke={strokeColor}
        strokeWidth={strokeWidth}
        cornerRadius={4}
        dash={damaged ? [6, 3] : undefined}
      />

      {/* Airlock indicator: small chevron on top-right corner */}
      {hullAirlock && type === "airlock" && (
        <Line
          points={[w - 10, 4, w - 4, 4, w - 4, 10]}
          stroke="#aaaaaa"
          strokeWidth={1.5}
          lineCap="round"
          lineJoin="round"
        />
      )}

      {/* Damage cross-hatching overlay */}
      {damaged && (
        <>
          <Line points={[4, 4, w - 4, h - 4]} stroke="#a0303066" strokeWidth={1} />
          <Line points={[w - 4, 4, 4, h - 4]} stroke="#a0303066" strokeWidth={1} />
        </>
      )}

      <Text
        x={4}
        y={h / 2 - 7}
        width={w - 8}
        text={damaged ? `[OFFLINE] ${label}` : label}
        fontSize={11}
        fill={damaged ? "#c07070" : "#f5f7ff"}
        align="center"
        wrap="none"
        ellipsis
      />
    </Group>
  );
}

export { SCALE };
