import type { RefObject } from "react";
import { Stage, Layer } from "react-konva";
import type Konva from "konva";
import type { StationLayout } from "../types";
import type { EditorAction } from "../editor/editorReducer";
import { RoomShape, SCALE } from "./RoomShape";
import { CorridorLine } from "./CorridorLine";

type Props = {
  layout: StationLayout;
  activeFloor: number;
  selectedId: string | null;
  connectingFromId: string | null;
  dispatch: (a: EditorAction) => void;
  stageRef: RefObject<Konva.Stage | null>;
  width: number;
  height: number;
};

export function FloorPlanCanvas({ layout, activeFloor, selectedId, connectingFromId, dispatch, stageRef, width, height }: Props) {
  const modules = layout.modules.filter(m => m.floor === activeFloor);
  const moduleIds = new Set(modules.map(m => m.id));
  const corridors = layout.corridors.filter(c => moduleIds.has(c.fromId) && moduleIds.has(c.toId));

  const canvasW = layout.bounds.w * SCALE;
  const canvasH = layout.bounds.h * SCALE;
  const fitScale = Math.min(width / canvasW, height / canvasH, 1);

  function handleModuleClick(id: string) {
    if (connectingFromId !== null && connectingFromId !== id) {
      dispatch({ type: "ADD_CORRIDOR", payload: { fromId: connectingFromId, toId: id, width: 1, waypoints: [] } });
      dispatch({ type: "SET_CONNECTING", payload: { id: null } });
      dispatch({ type: "SET_SELECTION", payload: { id } });
    } else {
      dispatch({ type: "SET_SELECTION", payload: { id: selectedId === id ? null : id } });
    }
  }

  return (
    <div style={{ cursor: connectingFromId ? "crosshair" : "default" }}>
      <Stage
        ref={stageRef}
        width={width}
        height={height}
        scaleX={fitScale}
        scaleY={fitScale}
        style={{ background: "#070b18", borderRadius: 8, border: "1px solid #24305b", display: "block" }}
        onClick={e => {
          if (e.target === e.target.getStage()) {
            dispatch({ type: "SET_SELECTION", payload: { id: null } });
          }
        }}
      >
        <Layer>
          {corridors.map(c => (
            <CorridorLine key={c.id} corridor={c} modules={modules} />
          ))}
        </Layer>
        <Layer>
          {modules.map(m => (
            <RoomShape
              key={m.id}
              module={m}
              selected={m.id === selectedId}
              onClick={() => handleModuleClick(m.id)}
              onDragEnd={(x, y) =>
                dispatch({ type: "MOVE_MODULE", payload: { id: m.id, rect: { ...m.rect, x, y } } })
              }
            />
          ))}
        </Layer>
      </Stage>
    </div>
  );
}
