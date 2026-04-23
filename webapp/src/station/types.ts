export type Vec2 = { x: number; y: number };

export type ModuleType =
  | "command"
  | "habitat"
  | "engineering"
  | "medical"
  | "cargo"
  | "airlock"
  | "lab"
  | "reactor"
  | "docking";

export type StationModule = {
  id: string;
  type: ModuleType;
  label: string;
  rect: { x: number; y: number; w: number; h: number };
  floor: number;
  damaged?: boolean;
  hullAirlock?: boolean;
  meta: Record<string, string | number>;
};

export type Corridor = {
  id: string;
  fromId: string;
  toId: string;
  width: number;
  waypoints: Vec2[];
};

export type StationLayout = {
  seed: string;
  params: GenerationParams;
  modules: StationModule[];
  corridors: Corridor[];
  bounds: { w: number; h: number };
};

export type GenerationParams = {
  seed: string;
  purpose: "military" | "research" | "civilian" | "mining" | "derelict";
  size: "small" | "medium" | "large" | "huge";
  moduleCount: number;
  symmetry: "none" | "bilateral" | "radial";
  density: number;
  floorCount: number;
  stationShape?: "ring" | "cylinder" | "sphere" | "box";
};

export const MODULE_COLORS: Record<ModuleType, string> = {
  command: "#3b6fd4",
  habitat: "#2e9e5b",
  engineering: "#c47d2a",
  medical: "#c43a3a",
  cargo: "#7a6aaa",
  airlock: "#888888",
  lab: "#2aadc4",
  reactor: "#d4a83b",
  docking: "#5a8a6a",
};

export const MODULE_HEIGHT: Record<ModuleType, number> = {
  command: 4,
  habitat: 3,
  engineering: 3,
  medical: 3,
  cargo: 2,
  airlock: 2,
  lab: 3,
  reactor: 4,
  docking: 2,
};
