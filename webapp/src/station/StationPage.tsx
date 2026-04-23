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
  floorCount: 2,
};

const SIDEBAR = 300;
const GAP = 16;

function useContainerWidth(ref: React.RefObject<HTMLDivElement | null>) {
  const [width, setWidth] = useState(900);
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
  const [activeFloor, setActiveFloor] = useState(0);
  const [showSkin, setShowSkin] = useState(false);
  const { state, dispatch } = useEditor();
  const stageRef = useRef<Konva.Stage>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const containerWidth = useContainerWidth(containerRef);
  const gltfExporterRef = useRef<(() => void) | null>(null);

  const canvasW = Math.max(360, containerWidth - SIDEBAR - GAP);
  const canvasH = Math.round(canvasW * 0.62);

  const handleGenerate = useCallback(() => {
    const seed = params.seed.trim() || Math.random().toString(36).slice(2, 10);
    const layout = generate({ ...params, seed });
    dispatch({ type: "LOAD_LAYOUT", payload: layout });
    setActiveFloor(0);
  }, [params, dispatch]);

  const handleExporterReady = useCallback((fn: () => void) => {
    gltfExporterRef.current = fn;
  }, []);

  const selectedModule = state.layout.modules.find(m => m.id === state.selectedId);
  const hasLayout = state.layout.modules.length > 0;
  const floorCount = state.layout.params.floorCount ?? 1;
  const floors = Array.from({ length: floorCount }, (_, i) => i);

  return (
    <div ref={containerRef} style={{ width: "min(1280px, 100%)", margin: "0 auto", padding: "1rem", boxSizing: "border-box" }}>
      <div style={{ display: "flex", gap: GAP, alignItems: "start" }}>

        {/* Sidebar */}
        <div style={{ width: SIDEBAR, flexShrink: 0, display: "flex", flexDirection: "column", gap: "1rem" }}>
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

        {/* Canvas area */}
        <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: "0.75rem" }}>

          {/* View + floor tabs */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap" }}>
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
              3D View
            </button>

            {hasLayout && view === "3d" && (
              <button
                className="button secondary"
                style={{ margin: 0, width: "auto", padding: "0.45rem 1rem", borderColor: showSkin ? "#8a97ab" : "#4a5a8a" }}
                onClick={() => setShowSkin(s => !s)}
              >
                {showSkin ? "Interior View" : "Exterior View"}
              </button>
            )}

            {hasLayout && view === "2d" && floorCount > 1 && (
              <div style={{ display: "flex", gap: "0.35rem", marginLeft: "0.5rem", borderLeft: "1px solid #24305b", paddingLeft: "0.75rem" }}>
                {floors.map(f => (
                  <button
                    key={f}
                    className={`button ${activeFloor === f ? "primary" : "secondary"}`}
                    style={{ margin: 0, width: "auto", padding: "0.35rem 0.75rem", fontSize: "0.85rem" }}
                    onClick={() => setActiveFloor(f)}
                  >
                    Deck {f + 1}
                  </button>
                ))}
              </div>
            )}
          </div>

          {!hasLayout && (
            <div className="card" style={{ textAlign: "center", padding: "3rem 1rem", minHeight: 300, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <p style={{ color: "#b4bee8", margin: 0 }}>
                Configure parameters and click <strong>Generate Station</strong> to create a floor plan.
              </p>
            </div>
          )}

          {hasLayout && view === "2d" && (
            <div>
              <FloorPlanCanvas
                layout={state.layout}
                activeFloor={activeFloor}
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
              showSkin={showSkin}
              dispatch={dispatch}
              width={canvasW}
              height={canvasH}
              onExporterReady={handleExporterReady}
            />
          )}

          {hasLayout && (
            <div className="card">
              <p className="meta" style={{ margin: 0 }}>
                <strong style={{ color: "#d3dcff" }}>Seed:</strong> {state.layout.seed}&nbsp; |&nbsp;
                <strong style={{ color: "#d3dcff" }}>Decks:</strong> {floorCount}&nbsp; |&nbsp;
                <strong style={{ color: "#d3dcff" }}>Modules:</strong> {state.layout.modules.length}&nbsp; |&nbsp;
                <strong style={{ color: "#d3dcff" }}>Corridors:</strong> {state.layout.corridors.length}&nbsp; |&nbsp;
                <strong style={{ color: "#d3dcff" }}>Purpose:</strong> {state.layout.params.purpose}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
