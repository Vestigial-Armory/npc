import { useEffect, useCallback } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { GLTFExporter } from "three/addons/exporters/GLTFExporter.js";
import type { StationLayout } from "../types";
import type { EditorAction } from "../editor/editorReducer";
import { RoomBox, FLOOR_GAP } from "./RoomBox";
import { CorridorTube } from "./CorridorTube";
import { HullSkin } from "./HullSkin";
import { Starfield } from "./Starfield";

type Props = {
  layout: StationLayout;
  selectedId: string | null;
  showSkin: boolean;
  dispatch: (a: EditorAction) => void;
  width: number;
  height: number;
  onExporterReady?: (fn: () => void) => void;
};

function GltfExporter({ onReady, seed }: { onReady?: (fn: () => void) => void; seed: string }) {
  const { scene } = useThree();
  const exportFn = useCallback(() => {
    const exporter = new GLTFExporter();
    exporter.parse(
      scene,
      result => {
        const blob = new Blob([result as ArrayBuffer], { type: "model/gltf-binary" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `station-${seed}.glb`;
        a.click();
        URL.revokeObjectURL(url);
      },
      err => console.error("GLTF export failed:", err),
      { binary: true }
    );
  }, [scene, seed]);
  useEffect(() => { onReady?.(exportFn); }, [onReady, exportFn]);
  return null;
}

// Resets camera to a clean position whenever showSkin toggles, so the scene
// is never occluded after switching between interior and exterior modes.
function CameraManager({ showSkin, camPos, targetY }: {
  showSkin: boolean;
  camPos: [number, number, number];
  targetY: number;
}) {
  const { camera } = useThree();
  useEffect(() => {
    camera.position.set(...camPos);
    camera.lookAt(0, targetY, 0);
    camera.updateMatrixWorld();
  }, [showSkin, camPos, targetY, camera]);

  return (
    <OrbitControls
      target={[0, targetY, 0]}
      enableDamping
      dampingFactor={0.08}
      maxPolarAngle={Math.PI / 1.05}
    />
  );
}

export function StationModel3D({ layout, selectedId, showSkin, dispatch, width, height, onExporterReady }: Props) {
  const floorCount = layout.params.floorCount ?? 1;
  const totalHeight = floorCount * FLOOR_GAP;
  const maxSpan = Math.max(layout.bounds.w, layout.bounds.h);
  const shape = layout.params.stationShape ?? "box";

  const exteriorR = shape !== "box"
    ? maxSpan * (shape === "ring" ? 0.46 : shape === "sphere" ? 0.42 : 0.30)
    : 0;
  const camDist = Math.max(maxSpan * 0.9 + totalHeight * 0.5, exteriorR * 3.8);
  const targetY = totalHeight * 0.4;

  // Camera starts slightly to the side and above, looking at station centre
  const camPos: [number, number, number] = [camDist * 0.45, camDist * 0.65, camDist * 0.8];

  // Interior lights live at world-space equivalents of station-footprint centre
  const cx = layout.bounds.w / 2;
  const cz = layout.bounds.h / 2;

  return (
    <div style={{ width, height, borderRadius: 8, overflow: "hidden", border: "1px solid #24305b" }}>
      <Canvas
        camera={{ position: camPos, fov: 45 }}
        onCreated={({ gl }) => gl.setClearColor("#000005")}
        style={{ background: "#000005" }}
        onClick={() => dispatch({ type: "SET_SELECTION", payload: { id: null } })}
      >
        <Starfield />

        {showSkin ? (
          <>
            <ambientLight intensity={0.12} color="#1a2050" />
            <directionalLight position={[200, 120, 80]} intensity={2.2} color="#fff8f0" />
            <directionalLight position={[-150, -40, -100]} intensity={0.18} color="#2030a0" />
          </>
        ) : (
          <>
            <ambientLight intensity={0.45} />
            <directionalLight position={[50, 80, 40]} intensity={1.4} />
            <pointLight position={[cx - layout.bounds.w / 2, totalHeight + 15, cz - layout.bounds.h / 2]} intensity={0.8} color="#a0b0ff" />
          </>
        )}

        <group position={[-cx, 0, -cz]}>
          {showSkin ? (
            <HullSkin layout={layout} />
          ) : (
            <>
              {layout.corridors.map(c => (
                <CorridorTube key={c.id} corridor={c} modules={layout.modules} />
              ))}
              {layout.modules.map(m => (
                <RoomBox
                  key={m.id}
                  module={m}
                  selected={m.id === selectedId}
                  onClick={() =>
                    dispatch({ type: "SET_SELECTION", payload: { id: selectedId === m.id ? null : m.id } })
                  }
                />
              ))}
            </>
          )}
        </group>

        <GltfExporter onReady={onExporterReady} seed={layout.seed} />
        <CameraManager showSkin={showSkin} camPos={camPos} targetY={targetY} />
      </Canvas>
    </div>
  );
}
