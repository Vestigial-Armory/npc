import { Rect, Text, Group } from "react-konva";
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
  const { rect, type, label } = module;
  const color = MODULE_COLORS[type];

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
        width={rect.w * SCALE}
        height={rect.h * SCALE}
        fill={color + "44"}
        stroke={selected ? "#ffffff" : color}
        strokeWidth={selected ? 2.5 : 1.5}
        cornerRadius={4}
      />
      <Text
        x={4}
        y={rect.h * SCALE / 2 - 7}
        width={rect.w * SCALE - 8}
        text={label}
        fontSize={11}
        fill="#f5f7ff"
        align="center"
        wrap="none"
        ellipsis
      />
    </Group>
  );
}

export { SCALE };
