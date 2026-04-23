import { useMemo } from "react";
import * as THREE from "three";
import type { StationLayout, StationModule, ModuleType } from "../types";
import { MODULE_COLORS } from "../types";
import { FLOOR_GAP } from "./RoomBox";

const BASE_HULL = new THREE.Color("#8a97ab");
const EXTERIOR_H = 5.5;
const MARGIN = 0.7;

const WINDOW_TYPES = new Set<ModuleType>(["habitat", "lab", "command", "medical"]);
const FIN_TYPES = new Set<ModuleType>(["reactor", "engineering"]);

function HullPanel({ module }: { module: StationModule }) {
  const { rect, floor, type, hullAirlock, damaged } = module;
  const yBase = floor * FLOOR_GAP;
  const w = rect.w + MARGIN * 2;
  const h = EXTERIOR_H;
  const d = rect.h + MARGIN * 2;
  const cx = rect.x + rect.w / 2;
  const cz = rect.y + rect.h / 2;

  const hullColor = useMemo(() => {
    const col = BASE_HULL.clone().lerp(new THREE.Color(MODULE_COLORS[type]), 0.07);
    if (damaged) col.lerp(new THREE.Color("#2a1a18"), 0.45);
    return col;
  }, [type, damaged]);

  const hullMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: hullColor,
    metalness: 0.62,
    roughness: 0.38,
    ...(damaged ? { transparent: true, opacity: 0.72 } : {}),
  }), [hullColor, damaged]);

  // Inset trim strip along the top edge to suggest panel layering
  const trimMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: new THREE.Color(hullColor).lerp(new THREE.Color("#000"), 0.25),
    metalness: 0.7,
    roughness: 0.3,
  }), [hullColor]);

  const windowCount = WINDOW_TYPES.has(type) ? Math.max(1, Math.floor(w / 5)) : 0;
  const windowMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: "#0d1f3c",
    emissive: "#1a3a70",
    emissiveIntensity: 0.9,
    roughness: 0.1,
    metalness: 0.2,
  }), []);

  const airlockMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: "#b87830",
    metalness: 0.55,
    roughness: 0.4,
  }), []);

  const finMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: "#5a1818",
    metalness: 0.3,
    roughness: 0.55,
    side: THREE.DoubleSide,
  }), []);

  return (
    <group position={[cx, yBase + h / 2, cz]}>
      {/* Main hull box */}
      <mesh>
        <boxGeometry args={[w, h, d]} />
        <primitive object={hullMat} attach="material" />
      </mesh>

      {/* Top trim panel */}
      <mesh position={[0, h / 2 - 0.15, 0]}>
        <boxGeometry args={[w + 0.05, 0.3, d + 0.05]} />
        <primitive object={trimMat} attach="material" />
      </mesh>

      {/* Bottom trim panel */}
      <mesh position={[0, -h / 2 + 0.15, 0]}>
        <boxGeometry args={[w + 0.05, 0.3, d + 0.05]} />
        <primitive object={trimMat} attach="material" />
      </mesh>

      {/* Windows along the longer axis */}
      {windowCount > 0 && (() => {
        const useLongAxis = w >= d;
        const span = useLongAxis ? w : d;
        const spacing = span / (windowCount + 1);
        return Array.from({ length: windowCount }, (_, i) => {
          const offset = -span / 2 + spacing * (i + 1);
          return (
            <group key={i}>
              {/* front face */}
              <mesh position={useLongAxis ? [offset, 0, -d / 2 - 0.06] : [-w / 2 - 0.06, 0, offset]}>
                <boxGeometry args={[1.1, 0.75, 0.12]} />
                <primitive object={windowMat} attach="material" />
              </mesh>
              {/* back face */}
              <mesh position={useLongAxis ? [offset, 0, d / 2 + 0.06] : [w / 2 + 0.06, 0, offset]}>
                <boxGeometry args={[1.1, 0.75, 0.12]} />
                <primitive object={windowMat} attach="material" />
              </mesh>
            </group>
          );
        });
      })()}

      {/* Airlock docking ring */}
      {hullAirlock && type === "airlock" && (
        <group position={[0, -h * 0.1, -d / 2 - 1.0]}>
          <mesh>
            <cylinderGeometry args={[1.1, 1.35, 1.8, 14]} />
            <primitive object={airlockMat} attach="material" />
          </mesh>
          {/* inner ring */}
          <mesh position={[0, 0.95, 0]}>
            <torusGeometry args={[1.0, 0.15, 8, 16]} />
            <primitive object={airlockMat} attach="material" />
          </mesh>
        </group>
      )}

      {/* Radiator fins (reactor / engineering) */}
      {FIN_TYPES.has(type) && (
        <>
          {/* fins along +Z side */}
          {[0.3, 0, -0.3].map((yo, k) => (
            <mesh key={`fz${k}`} position={[0, yo * h, d / 2 + 2.0]}>
              <boxGeometry args={[w * 0.75, 2.6, 0.14]} />
              <primitive object={finMat} attach="material" />
            </mesh>
          ))}
          {/* fins along -Z side */}
          {[0.3, 0, -0.3].map((yo, k) => (
            <mesh key={`fbz${k}`} position={[0, yo * h, -d / 2 - 2.0]}>
              <boxGeometry args={[w * 0.75, 2.6, 0.14]} />
              <primitive object={finMat} attach="material" />
            </mesh>
          ))}
        </>
      )}
    </group>
  );
}

export function HullSkin({ layout }: { layout: StationLayout }) {
  return (
    <group>
      {layout.modules.map(m => (
        <HullPanel key={m.id} module={m} />
      ))}
    </group>
  );
}
