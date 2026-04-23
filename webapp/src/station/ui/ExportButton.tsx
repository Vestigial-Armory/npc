import type { StationLayout } from "../types";

type Props = {
  layout: StationLayout | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  stageRef: React.RefObject<any>;
};

export function ExportButton({ layout, stageRef }: Props) {
  function exportPng() {
    if (!stageRef.current) return;
    const dataUrl = stageRef.current.toDataURL({ pixelRatio: 2 });
    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = `station-${layout?.seed ?? "export"}.png`;
    a.click();
  }

  function exportJson() {
    if (!layout) return;
    const blob = new Blob([JSON.stringify(layout, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `station-${layout.seed}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div style={{ display: "flex", gap: "0.5rem" }}>
      <button className="button secondary" style={{ margin: 0 }} onClick={exportPng} disabled={!layout}>
        Export PNG
      </button>
      <button className="button secondary" style={{ margin: 0 }} onClick={exportJson} disabled={!layout}>
        Export JSON
      </button>
    </div>
  );
}
