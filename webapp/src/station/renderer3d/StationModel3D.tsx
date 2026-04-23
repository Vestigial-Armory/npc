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

export function StationModel3D({ layout, selectedId, showSkin, dispatch, width, height, onExporterReady }: Props) {
  const cx = layout.bounds.w / 2;
  const cz = layout.bounds.h / 2;
  const floorCount = layout.params.floorCount ?? 1;
  const totalHeight = floorCount * FLOOR_GAP;
  const camDist = Math.max(layout.bounds.w, layout.bounds.h) * 0.9 + totalHeight * 0.5;

  // Exterior view: sun-like directional light from one side, dim fill from the other
  // Interior view: warmer overhead point lights
  return (
    <div style={{ width, height, borderRadius: 8, overflow: "hidden", border: "1px solid #24305b" }}>
      <Canvas
        camera={{ position: [cx, camDist * 0.7, cz + camDist * 0.6], fov: 45 }}
        onCreated={({ gl }) => gl.setClearColor("#000005")}
        style={{ background: "#000005" }}
        onClick={() => dispatch({ type: "SET_SELECTION", payload: { id: null } })}
      >
        <Starfield />

        {showSkin ? (
          <>
            {/* Exterior lighting: harsh directional sun, soft blue fill */}
            <ambientLight intensity={0.12} color="#1a2050" />
            <directionalLight position={[200, 120, 80]} intensity={2.2} color="#fff8f0" />
            <directionalLight position={[-150, -40, -100]} intensity={0.18} color="#2030a0" />
          </>
        ) : (
          <>
            {/* Interior lighting: warm overhead, soft ambient */}
            <ambientLight intensity={0.35} />
            <directionalLight position={[cx + 40, 80, cz + 30]} intensity={1.3} castShadow />
            <pointLight position={[cx, totalHeight + 20, cz]} intensity={0.7} color="#a0b0ff" />
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

        <OrbitControls
          target={[0, totalHeight * 0.4, 0]}
          enableDamping
          dampingFactor={0.08}
        />
      </Canvas>
    </div>
  );
}
