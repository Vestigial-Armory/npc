import { Ring, Circle, Line } from "react-konva";
import type { StationLayout } from "../types";
import { SCALE } from "./RoomShape";

type Props = { layout: StationLayout };

export function ShapeOutline({ layout }: Props) {
  const shape = layout.params.stationShape ?? "box";
  if (shape === "box") return null;

  const cx   = (layout.bounds.w / 2) * SCALE;
  const cy   = (layout.bounds.h / 2) * SCALE;
  const maxR = Math.min(layout.bounds.w / 2, layout.bounds.h / 2);

  const strokeProps = { stroke: "#2a3a60", strokeWidth: 1.5, listening: false };
  const guideProps  = { stroke: "#141e3a", strokeWidth: 1, dash: [5, 5], listening: false };

  if (shape === "ring") {
    const outerR = maxR * 0.82 * SCALE;
    const ringW  = outerR * 0.30;
    const innerR = outerR - ringW;
    const hubR   = (maxR * 0.82 * 0.30 * 0.62 / 2) * SCALE;

    // Guide lines: radial spokes
    const spokeCount = 12;
    const spokes = Array.from({ length: spokeCount }, (_, i) => {
      const angle = (i / spokeCount) * Math.PI * 2;
      return [
        cx + innerR * Math.cos(angle),
        cy + innerR * Math.sin(angle),
        cx + outerR * Math.cos(angle),
        cy + outerR * Math.sin(angle),
      ];
    });

    return (
      <>
        {/* Ring footprint fill */}
        <Ring x={cx} y={cy} innerRadius={innerR} outerRadius={outerR} fill="#080d1a" {...strokeProps} />
        {/* Hub footprint */}
        <Circle x={cx} y={cy} radius={hubR} fill="#080d1a" {...strokeProps} />
        {/* Radial guide lines */}
        {spokes.map((pts, i) => (
          <Line key={i} points={pts} {...guideProps} />
        ))}
      </>
    );
  }

  if (shape === "sphere" || shape === "cylinder") {
    const outerR = maxR * 0.84 * SCALE;
    const innerR = outerR * 0.44;
    const hubR   = maxR * 0.20 * 0.5 * SCALE;

    // Guide lines: radial spokes
    const spokeCount = 8;
    const spokes = Array.from({ length: spokeCount }, (_, i) => {
      const angle = (i / spokeCount) * Math.PI * 2;
      return [
        cx + hubR * Math.cos(angle),
        cy + hubR * Math.sin(angle),
        cx + outerR * Math.cos(angle),
        cy + outerR * Math.sin(angle),
      ];
    });

    return (
      <>
        {/* Outer disc */}
        <Circle x={cx} y={cy} radius={outerR} fill="#080d1a" {...strokeProps} />
        {/* Inner ring guide */}
        <Circle x={cx} y={cy} radius={innerR} fill="transparent" {...guideProps} />
        {/* Hub */}
        <Circle x={cx} y={cy} radius={hubR} fill="#080d1a" {...strokeProps} />
        {/* Spokes */}
        {spokes.map((pts, i) => (
          <Line key={i} points={pts} {...guideProps} />
        ))}
      </>
    );
  }

  return null;
}
