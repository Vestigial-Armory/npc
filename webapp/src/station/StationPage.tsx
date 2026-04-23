import { useState, useRef, useCallback, useEffect } from "react";
import type Konva from "konva";
import type { GenerationParams } from "./types";
import { generate } from "./generator/generate";
import { useEditor } from "./editor/useEditor";
import { FloorPlanCanvas } from "./renderer2d/FloorPlanCanvas";
import { StationModel3D } from "./renderer3d/StationModel3D";
import { ParamsForm } from "./ui/ParamsForm";
import { EditPanel } from "./ui/EditPanel";
import { ExportButton } from "./ui/ExportButton";
import { Legend } from "./renderer2d/Legend";

const DEFAULT_PARAMS: GenerationParams = {
  seed: "station-alpha",
  purpose: "research",
  size: "medium",
  moduleCount: 12,
  symmetry: "none",
  density: 0.3,
};

function useContainerWidth(ref: React.RefObject<HTMLDivElement | null>) {
  const [width, setWidth] = useState(800);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(entries => setWidth(entries[0].contentRect.width));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, [ref]);
  return width;
}

export function StationPage() {
  const [params, setParams] = useState<GenerationParams>(DEFAULT_PARAMS);
  const [view, setView] = useState<"2d" | "3d">("2d");
  const { state, dispatch } = useEditor();
  const stageRef = useRef<Konva.Stage>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const containerWidth = useContainerWidth(containerRef);
  const gltfExporterRef = useRef<(() => void) | null>(null);

  const narrow = containerWidth < 720;
  const SIDEBAR = 300;
  const GAP = 16;
  const canvasW = narrow ? containerWidth : Math.max(300, containerWidth - SIDEBAR - GAP);
  const canvasH = Math.round(canvasW * 0.6);

  const handleGenerate = useCallback(() => {
    const seed = params.seed.trim() || Math.random().toString(36).slice(2, 10);
    const layout = generate({ ...params, seed });
    dispatch({ type: "LOAD_LAYOUT", payload: layout });
  }, [params, dispatch]);

  const handleExporterReady = useCallback((fn: () => void) => {
    gltfExporterRef.current = fn;
  }, []);

  const selectedModule = state.layout.modules.find(m => m.id === state.selectedId);
  const hasLayout = state.layout.modules.length > 0;

  const sidebar = (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem", width: narrow ? "100%" : SIDEBAR, flexShrink: 0 }}>
      <ParamsForm params={params} onChange={setParams} onGenerate={handleGenerate} />
      <EditPanel module={selectedModule} connectingFromId={state.connectingFromId} dispatch={dispatch} />
      {hasLayout && (
        <div className="card">
          <span className="label" style={{ marginTop: 0 }}>Export</span>
          <div style={{ marginTop: "0.5rem" }}>
            <ExportButton layout={state.layout} stageRef={stageRef} gltfExporter={gltfExporterRef.current} />
          </div>
        </div>
      )}
    </div>
  );

  const canvas = (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem", minWidth: 0 }}>
      <div style={{ display: "flex", gap: "0.5rem" }}>
        <button
          className={`button ${view === "2d" ? "primary" : "secondary"}`}
          style={{ margin: 0, width: "auto", padding: "0.45rem 1rem" }}
          onClick={() => setView("2d")}
        >
          2D Floor Plan
        </button>
        <button
          className={`button ${view === "3d" ? "primary" : "secondary"}`}
          style={{ margin: 0, width: "auto", padding: "0.45rem 1rem" }}
          onClick={() => setView("3d")}
        >
          3D Model
        </button>
      </div>

      {!hasLayout && (
        <div className="card" style={{ textAlign: "center", padding: "3rem 1rem" }}>
          <p style={{ color: "#b4bee8", margin: 0 }}>
            Configure parameters and click <strong>Generate Station</strong> to create a floor plan.
          </p>
        </div>
      )}

      {hasLayout && view === "2d" && (
        <div>
          <FloorPlanCanvas
            layout={state.layout}
            selectedId={state.selectedId}
            connectingFromId={state.connectingFromId}
            dispatch={dispatch}
            stageRef={stageRef}
            width={canvasW}
            height={canvasH}
          />
          <Legend />
        </div>
      )}

      {hasLayout && view === "3d" && (
        <StationModel3D
          layout={state.layout}
          selectedId={state.selectedId}
          dispatch={dispatch}
          width={canvasW}
          height={canvasH}
          onExporterReady={handleExporterReady}
        />
      )}

      {hasLayout && (
        <div className="card">
          <p className="meta" style={{ margin: 0, wordBreak: "break-all" }}>
            <strong style={{ color: "#d3dcff" }}>Seed:</strong> {state.layout.seed} &nbsp;|&nbsp;
            <strong style={{ color: "#d3dcff" }}>Modules:</strong> {state.layout.modules.length} &nbsp;|&nbsp;
            <strong style={{ color: "#d3dcff" }}>Corridors:</strong> {state.layout.corridors.length} &nbsp;|&nbsp;
            <strong style={{ color: "#d3dcff" }}>Purpose:</strong> {state.layout.params.purpose}
          </p>
        </div>
      )}
    </div>
  );

  return (
    <div ref={containerRef} style={{ width: "min(1200px, 100%)", margin: "0 auto", padding: "1rem", boxSizing: "border-box" }}>
      <div style={{
        display: "flex",
        flexDirection: narrow ? "column" : "row",
        gap: "1rem",
        alignItems: "start",
      }}>
        {narrow ? (
          <>
            {canvas}
            {sidebar}
          </>
        ) : (
          <>
            {sidebar}
            {canvas}
          </>
        )}
      </div>
    </div>
  );
}
