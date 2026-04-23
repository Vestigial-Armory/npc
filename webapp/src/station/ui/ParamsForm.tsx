import type { GenerationParams } from "../types";

type Props = {
  params: GenerationParams;
  onChange: (p: GenerationParams) => void;
  onGenerate: () => void;
  disabled?: boolean;
};

export function ParamsForm({ params, onChange, onGenerate, disabled }: Props) {
  function set<K extends keyof GenerationParams>(key: K, value: GenerationParams[K]) {
    onChange({ ...params, [key]: value });
  }

  function randomizeSeed() {
    set("seed", Math.random().toString(36).slice(2, 10));
  }

  return (
    <div className="card">
      <h2 style={{ marginBottom: "0.25rem" }}>Station Generator</h2>
      <p className="subtitle" style={{ marginBottom: "0.75rem" }}>Procedural space station floor plans</p>

      <label className="label">Seed
        <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.35rem" }}>
          <input
            type="text"
            value={params.seed}
            onChange={e => set("seed", e.target.value)}
            placeholder="any text or leave blank"
            style={{ flex: 1 }}
          />
          <button className="button secondary" style={{ width: "auto", marginTop: 0, whiteSpace: "nowrap", padding: "0.5rem 0.75rem" }} onClick={randomizeSeed}>
            Random
          </button>
        </div>
      </label>

      <label className="label">Purpose
        <select value={params.purpose} onChange={e => set("purpose", e.target.value as GenerationParams["purpose"])} style={{ marginTop: "0.35rem" }}>
          <option value="research">Research</option>
          <option value="military">Military</option>
          <option value="civilian">Civilian</option>
          <option value="mining">Mining</option>
          <option value="derelict">Derelict</option>
        </select>
      </label>

      <label className="label">Size
        <select value={params.size} onChange={e => set("size", e.target.value as GenerationParams["size"])} style={{ marginTop: "0.35rem" }}>
          <option value="small">Small (4–8 modules)</option>
          <option value="medium">Medium (8–16 modules)</option>
          <option value="large">Large (14–24 modules)</option>
          <option value="huge">Huge (20–36 modules)</option>
        </select>
      </label>

      <label className="label">Module Count: {params.moduleCount}
        <input
          type="range"
          min={4}
          max={36}
          value={params.moduleCount}
          onChange={e => set("moduleCount", Number(e.target.value))}
          style={{ width: "100%", marginTop: "0.35rem", accentColor: "#3150d9" }}
        />
      </label>

      <label className="label">Symmetry
        <select value={params.symmetry} onChange={e => set("symmetry", e.target.value as GenerationParams["symmetry"])} style={{ marginTop: "0.35rem" }}>
          <option value="none">None</option>
          <option value="bilateral">Bilateral (mirrored)</option>
          <option value="radial">Radial (4-fold)</option>
        </select>
      </label>

      <label className="label">Corridor Density: {Math.round(params.density * 100)}%
        <input
          type="range"
          min={0}
          max={100}
          value={Math.round(params.density * 100)}
          onChange={e => set("density", Number(e.target.value) / 100)}
          style={{ width: "100%", marginTop: "0.35rem", accentColor: "#3150d9" }}
        />
      </label>

      <button className="button primary" style={{ marginTop: "1rem" }} disabled={disabled} onClick={onGenerate}>
        Generate Station
      </button>
    </div>
  );
}
