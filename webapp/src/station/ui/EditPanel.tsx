import type { StationModule, ModuleType } from "../types";
import { MODULE_COLORS } from "../types";
import type { EditorAction } from "../editor/editorReducer";
import { MODULE_LABELS } from "../generator/archetypes";

type Props = {
  module: StationModule | undefined;
  connectingFromId: string | null;
  dispatch: (a: EditorAction) => void;
};

const ALL_TYPES: ModuleType[] = [
  "command", "habitat", "engineering", "medical",
  "cargo", "airlock", "lab", "reactor", "docking",
];

export function EditPanel({ module, connectingFromId, dispatch }: Props) {
  if (connectingFromId !== null) {
    return (
      <div className="card">
        <p style={{ color: "#e2e9ff", margin: 0 }}>Click another module to connect with a corridor.</p>
        <button className="button secondary" onClick={() => dispatch({ type: "SET_CONNECTING", payload: { id: null } })}>
          Cancel
        </button>
      </div>
    );
  }

  if (!module) {
    return (
      <div className="card">
        <p className="meta">Select a module on the floor plan to edit it.</p>
      </div>
    );
  }

  function update(changes: Partial<Pick<StationModule, "type" | "label">>) {
    if (!module) return;
    dispatch({ type: "UPDATE_MODULE", payload: { id: module.id, changes } });
  }

  return (
    <div className="card">
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.5rem" }}>
        <div style={{ width: 14, height: 14, borderRadius: 3, background: MODULE_COLORS[module.type], flexShrink: 0 }} />
        <h2 style={{ margin: 0, fontSize: "1rem" }}>{module.label}</h2>
      </div>
      <p className="meta" style={{ marginTop: 0 }}>
        Position: ({module.rect.x}, {module.rect.y}) &nbsp; Size: {module.rect.w}×{module.rect.h}
      </p>

      <label className="label">Label
        <input
          type="text"
          value={module.label}
          onChange={e => update({ label: e.target.value })}
          style={{ marginTop: "0.35rem" }}
        />
      </label>

      <label className="label">Type
        <select
          value={module.type}
          onChange={e => {
            const type = e.target.value as ModuleType;
            const label = MODULE_LABELS[type][0];
            update({ type, label });
          }}
          style={{ marginTop: "0.35rem" }}
        >
          {ALL_TYPES.map(t => (
            <option key={t} value={t}>{t.charAt(0).toUpperCase() + t.slice(1)}</option>
          ))}
        </select>
      </label>

      <button
        className="button secondary"
        style={{ marginTop: "0.75rem" }}
        onClick={() => dispatch({ type: "SET_CONNECTING", payload: { id: module.id } })}
      >
        Add Corridor From This Module
      </button>

      <button
        className="button"
        style={{ marginTop: "0.5rem", background: "#3a1a1a", borderColor: "#7a2a2a", color: "#ffb9b9" }}
        onClick={() => dispatch({ type: "REMOVE_MODULE", payload: { id: module.id } })}
      >
        Delete Module
      </button>
    </div>
  );
}
