import type { ModuleType } from "../types";
import { MODULE_COLORS } from "../types";

const ALL_TYPES: ModuleType[] = [
  "command", "habitat", "engineering", "medical",
  "cargo", "airlock", "lab", "reactor", "docking",
];

const LABELS: Record<ModuleType, string> = {
  command: "Command", habitat: "Habitat", engineering: "Engineering",
  medical: "Medical", cargo: "Cargo", airlock: "Airlock",
  lab: "Lab", reactor: "Reactor", docking: "Docking",
};

export function Legend() {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem 0.8rem", marginTop: "0.5rem" }}>
      {ALL_TYPES.map(t => (
        <div key={t} style={{ display: "flex", alignItems: "center", gap: "0.35rem", fontSize: "0.8rem", color: "#b4bee8" }}>
          <div style={{ width: 12, height: 12, borderRadius: 3, background: MODULE_COLORS[t], border: `1px solid ${MODULE_COLORS[t]}` }} />
          {LABELS[t]}
        </div>
      ))}
    </div>
  );
}
