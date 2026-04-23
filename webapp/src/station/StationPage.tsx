import { useState, useRef, useCallback } from "react";
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

export function StationPage() {
  const [params, setParams] = useState<GenerationParams>(DEFAULT_PARAMS);
  const [view, setView] = useState<"2d" | "3d">("2d");
  const { state, dispatch } = useEditor();
  const stageRef = useRef<Konva.Stage>(null);

  const handleGenerate = useCallback(() => {
    const seed = params.seed.trim() || Math.random().toString(36).slice(2, 10);
    const layout = generate({ ...params, seed });
    dispatch({ type: "LOAD_LAYOUT", payload: layout });
  }, [params, dispatch]);

  const selectedModule = state.layout.modules.find(m => m.id === state.selectedId);
  const hasLayout = state.layout.modules.length > 0;

  return (
    <div style={{ width: "min(1200px, 100%)", margin: "0 auto", padding: "1rem", boxSizing: "border-box" }}>
      <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: "1rem", alignItems: "start" }}>

        {/* Left sidebar */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <ParamsForm
            params={params}
            onChange={setParams}
            onGenerate={handleGenerate}
          />
          <EditPanel
            module={selectedModule}
            connectingFromId={state.connectingFromId}
            dispatch={dispatch}
          />
          {hasLayout && (
            <div className="card">
              <ExportButton layout={state.layout} stageRef={stageRef} />
            </div>
          )}
        </div>

        {/* Main canvas area */}
        <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
          {/* View toggle */}
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
                width={Math.min(860, window.innerWidth - 340)}
                height={560}
              />
              <Legend />
            </div>
          )}

          {hasLayout && view === "3d" && (
            <StationModel3D
              layout={state.layout}
              selectedId={state.selectedId}
              dispatch={dispatch}
              width={Math.min(860, window.innerWidth - 340)}
              height={560}
            />
          )}

          {hasLayout && (
            <div className="card">
              <p className="meta" style={{ margin: 0 }}>
                <strong style={{ color: "#d3dcff" }}>Seed:</strong> {state.layout.seed} &nbsp;|&nbsp;
                <strong style={{ color: "#d3dcff" }}>Modules:</strong> {state.layout.modules.length} &nbsp;|&nbsp;
                <strong style={{ color: "#d3dcff" }}>Corridors:</strong> {state.layout.corridors.length} &nbsp;|&nbsp;
                <strong style={{ color: "#d3dcff" }}>Purpose:</strong> {state.layout.params.purpose}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
