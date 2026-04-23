import type { ModuleType, GenerationParams } from "../types";

type WeightMap = Record<ModuleType, number>;

const PURPOSE_WEIGHTS: Record<GenerationParams["purpose"], WeightMap> = {
  research: {
    command: 1, habitat: 2, engineering: 2, medical: 2,
    cargo: 1, airlock: 1, lab: 5, reactor: 1, docking: 1,
  },
  military: {
    command: 3, habitat: 2, engineering: 3, medical: 2,
    cargo: 2, airlock: 2, lab: 1, reactor: 2, docking: 2,
  },
  civilian: {
    command: 1, habitat: 5, engineering: 1, medical: 2,
    cargo: 2, airlock: 1, lab: 1, reactor: 1, docking: 3,
  },
  mining: {
    command: 1, habitat: 2, engineering: 3, medical: 1,
    cargo: 4, airlock: 1, lab: 1, reactor: 3, docking: 2,
  },
  derelict: {
    command: 1, habitat: 2, engineering: 1, medical: 1,
    cargo: 2, airlock: 2, lab: 1, reactor: 1, docking: 1,
  },
};

export function weightedPickType(
  purpose: GenerationParams["purpose"],
  rngFloat: () => number
): ModuleType {
  const weights = PURPOSE_WEIGHTS[purpose];
  const types = Object.keys(weights) as ModuleType[];
  const total = types.reduce((s, t) => s + weights[t], 0);
  let r = rngFloat() * total;
  for (const t of types) {
    r -= weights[t];
    if (r <= 0) return t;
  }
  return types[types.length - 1];
}

export const MODULE_LABELS: Record<ModuleType, string[]> = {
  command:     ["Command Deck", "Bridge", "Operations Center", "Command Hub"],
  habitat:     ["Crew Quarters", "Hab Block", "Living Module", "Barracks"],
  engineering: ["Engineering Bay", "Systems Room", "Maintenance Bay", "Tech Hub"],
  medical:     ["Medical Bay", "Infirmary", "Sick Bay", "Med Center"],
  cargo:       ["Cargo Hold", "Storage Bay", "Supply Depot", "Warehouse"],
  airlock:     ["Airlock", "EVA Lock", "External Access", "Pressure Lock"],
  lab:         ["Research Lab", "Science Bay", "Analysis Suite", "Lab Module"],
  reactor:     ["Reactor Core", "Power Plant", "Fusion Core", "Energy Module"],
  docking:     ["Docking Bay", "Hangar", "Berthing Module", "Dock Alpha"],
};
