import { useReducer } from "react";
import { editorReducer, type EditorState, type EditorAction } from "./editorReducer";
import type { StationLayout } from "../types";

const EMPTY_LAYOUT: StationLayout = {
  seed: "",
  params: {
    seed: "",
    purpose: "research",
    size: "medium",
    moduleCount: 10,
    symmetry: "none",
    density: 0.3,
  },
  modules: [],
  corridors: [],
  bounds: { w: 70, h: 50 },
};

export function useEditor(initial?: StationLayout) {
  const [state, dispatch] = useReducer<EditorState, [EditorAction]>(
    editorReducer,
    { layout: initial ?? EMPTY_LAYOUT, selectedId: null, connectingFromId: null }
  );
  return { state, dispatch };
}
