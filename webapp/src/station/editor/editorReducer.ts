import type { StationLayout, StationModule, Corridor } from "../types";

export type EditorState = {
  layout: StationLayout;
  selectedId: string | null;
  connectingFromId: string | null;
};

export type EditorAction =
  | { type: "LOAD_LAYOUT"; payload: StationLayout }
  | { type: "ADD_MODULE"; payload: Omit<StationModule, "id"> }
  | { type: "REMOVE_MODULE"; payload: { id: string } }
  | { type: "MOVE_MODULE"; payload: { id: string; rect: StationModule["rect"] } }
  | { type: "UPDATE_MODULE"; payload: { id: string; changes: Partial<Pick<StationModule, "type" | "label" | "meta">> } }
  | { type: "ADD_CORRIDOR"; payload: Omit<Corridor, "id"> }
  | { type: "REMOVE_CORRIDOR"; payload: { id: string } }
  | { type: "SET_SELECTION"; payload: { id: string | null } }
  | { type: "SET_CONNECTING"; payload: { id: string | null } };

let idCounter = 1000;
function nextId(prefix: string) {
  return `${prefix}${++idCounter}`;
}

export function editorReducer(state: EditorState, action: EditorAction): EditorState {
  switch (action.type) {
    case "LOAD_LAYOUT":
      return { layout: action.payload, selectedId: null, connectingFromId: null };

    case "ADD_MODULE": {
      const mod: StationModule = { ...action.payload, id: nextId("m") };
      return {
        ...state,
        layout: { ...state.layout, modules: [...state.layout.modules, mod] },
      };
    }

    case "REMOVE_MODULE": {
      const { id } = action.payload;
      return {
        ...state,
        selectedId: state.selectedId === id ? null : state.selectedId,
        layout: {
          ...state.layout,
          modules: state.layout.modules.filter(m => m.id !== id),
          corridors: state.layout.corridors.filter(c => c.fromId !== id && c.toId !== id),
        },
      };
    }

    case "MOVE_MODULE":
      return {
        ...state,
        layout: {
          ...state.layout,
          modules: state.layout.modules.map(m =>
            m.id === action.payload.id ? { ...m, rect: action.payload.rect } : m
          ),
        },
      };

    case "UPDATE_MODULE":
      return {
        ...state,
        layout: {
          ...state.layout,
          modules: state.layout.modules.map(m =>
            m.id === action.payload.id ? { ...m, ...action.payload.changes } : m
          ),
        },
      };

    case "ADD_CORRIDOR": {
      const corridor: Corridor = { ...action.payload, id: nextId("c") };
      return {
        ...state,
        layout: { ...state.layout, corridors: [...state.layout.corridors, corridor] },
      };
    }

    case "REMOVE_CORRIDOR":
      return {
        ...state,
        layout: {
          ...state.layout,
          corridors: state.layout.corridors.filter(c => c.id !== action.payload.id),
        },
      };

    case "SET_SELECTION":
      return { ...state, selectedId: action.payload.id };

    case "SET_CONNECTING":
      return { ...state, connectingFromId: action.payload.id };

    default:
      return state;
  }
}
