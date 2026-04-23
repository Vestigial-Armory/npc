import { useEffect, useCallback } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, Grid } from "@react-three/drei";
import { GLTFExporter } from "three/addons/exporters/GLTFExporter.js";
import type { StationLayout } from "../types";
import type { EditorAction } from "../editor/editorReducer";
import { RoomBox } from "./RoomBox";
import { CorridorTube } from "./CorridorTube";

type Props = {
  layout: StationLayout;
  selectedId: string | null;
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

export function StationModel3D({ layout, selectedId, dispatch, width, height, onExporterReady }: Props) {
  const cx = layout.bounds.w / 2;
  const cz = layout.bounds.h / 2;

  return (
    <div style={{ width, height, borderRadius: 8, overflow: "hidden", border: "1px solid #24305b" }}>
      <Canvas
        camera={{ position: [cx, Math.max(layout.bounds.w, layout.bounds.h) * 0.9, cz + 20], fov: 45 }}
        style={{ background: "#070b18" }}
        onClick={() => dispatch({ type: "SET_SELECTION", payload: { id: null } })}
      >
        <ambientLight intensity={0.4} />
        <directionalLight position={[cx + 30, 60, cz + 20]} intensity={1.2} castShadow />
        <pointLight position={[cx, 20, cz]} intensity={0.6} color="#a0b0ff" />

        <group position={[-cx, 0, -cz]}>
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
          <Grid
            position={[cx, -0.05, cz]}
            args={[layout.bounds.w * 1.2, layout.bounds.h * 1.2]}
            cellSize={10}
            cellColor="#1a2550"
            sectionSize={20}
            sectionColor="#2a3a70"
            fadeDistance={200}
            infiniteGrid
          />
        </group>

        <GltfExporter onReady={onExporterReady} seed={layout.seed} />

        <OrbitControls
          target={[0, 2, 0]}
          maxPolarAngle={Math.PI / 2.05}
          enableDamping
          dampingFactor={0.08}
        />
      </Canvas>
    </div>
  );
}
