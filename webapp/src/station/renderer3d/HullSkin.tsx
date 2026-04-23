import { useMemo } from "react";
import * as THREE from "three";
import type { StationLayout, StationModule, ModuleType } from "../types";
import { MODULE_COLORS } from "../types";
import { FLOOR_GAP } from "./RoomBox";

// ── Shared material helpers ────────────────────────────────────────────

function mkHull(tint?: string) {
  const col = new THREE.Color("#8e9daf");
  if (tint) col.lerp(new THREE.Color(tint), 0.12);
  return new THREE.MeshStandardMaterial({ color: col, metalness: 0.65, roughness: 0.35 });
}
function mkAccent() {
  return new THREE.MeshStandardMaterial({ color: "#627080", metalness: 0.72, roughness: 0.28 });
}
function mkWindow() {
  return new THREE.MeshStandardMaterial({
    color: "#0c1a30", emissive: "#2a55c0", emissiveIntensity: 2.0, roughness: 0.08, metalness: 0.1,
  });
}
function mkSolar() {
  return new THREE.MeshStandardMaterial({
    color: "#152a60", emissive: "#1a3a80", emissiveIntensity: 0.4, metalness: 0.35, roughness: 0.5, side: THREE.DoubleSide,
  });
}
function mkFin() {
  return new THREE.MeshStandardMaterial({ color: "#561414", metalness: 0.3, roughness: 0.6, side: THREE.DoubleSide });
}
function mkDock() {
  return new THREE.MeshStandardMaterial({ color: "#b07828", metalness: 0.55, roughness: 0.4 });
}

// ── Ring station ───────────────────────────────────────────────────────

function Pylon({ angle, hubR, ringR, sweepY }: { angle: number; hubR: number; ringR: number; sweepY: number }) {
  const curve = useMemo(() => new THREE.CatmullRomCurve3([
    new THREE.Vector3(hubR * 1.15 * Math.cos(angle), 0,           hubR * 1.15 * Math.sin(angle)),
    new THREE.Vector3(ringR * 0.33 * Math.cos(angle), sweepY,      ringR * 0.33 * Math.sin(angle)),
    new THREE.Vector3(ringR * 0.70 * Math.cos(angle), sweepY * 0.3, ringR * 0.70 * Math.sin(angle)),
    new THREE.Vector3(ringR        * Math.cos(angle), 0,            ringR        * Math.sin(angle)),
  ]), [angle, hubR, ringR, sweepY]);

  const geo  = useMemo(() => new THREE.TubeGeometry(curve, 30, hubR * 0.14, 9, false), [curve, hubR]);
  const mat  = useMemo(mkHull, []);
  return <mesh geometry={geo} material={mat} />;
}

function RingExterior({ r }: { r: number }) {
  const tube   = r * 0.13;
  const hubR   = r * 0.115;
  const hubH   = r * 0.30;
  const innerR = r * 0.58;
  const innerT = tube * 0.52;

  const mat    = useMemo(mkHull, []);
  const acc    = useMemo(mkAccent, []);
  const win    = useMemo(mkWindow, []);
  const dock   = useMemo(mkDock, []);

  const pylons = useMemo(() =>
    Array.from({ length: 6 }, (_, i) => ({
      angle:  (i / 6) * Math.PI * 2,
      sweepY: i % 2 === 0 ? r * 0.40 : -r * 0.22,
    })), [r]);

  return (
    <group>
      {/* ── Outer docking ring ── */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[r, tube, 18, 100]} />
        <primitive object={mat} attach="material" />
      </mesh>
      {/* Window strip on inner face of outer ring */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[r - tube * 0.42, 0.48, 4, 140]} />
        <primitive object={win} attach="material" />
      </mesh>

      {/* ── Inner habitat ring ── */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[innerR, innerT, 14, 72]} />
        <primitive object={acc} attach="material" />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[innerR - innerT * 0.38, 0.3, 4, 100]} />
        <primitive object={win} attach="material" />
      </mesh>

      {/* ── Radial spokes connecting inner to outer ring (12 thin) ── */}
      {Array.from({ length: 12 }, (_, i) => {
        const a   = (i / 12) * Math.PI * 2;
        const len = r - innerR;
        const midX = Math.cos(a) * (innerR + len / 2);
        const midZ = Math.sin(a) * (innerR + len / 2);
        return (
          <mesh key={i} position={[midX, 0, midZ]} rotation={[0, -a, Math.PI / 2]}>
            <cylinderGeometry args={[0.35, 0.35, len, 6]} />
            <primitive object={acc} attach="material" />
          </mesh>
        );
      })}

      {/* ── Central command hub ── */}
      <mesh>
        <cylinderGeometry args={[hubR, hubR * 0.88, hubH, 22]} />
        <primitive object={mat} attach="material" />
      </mesh>
      {/* Hub top dome */}
      <mesh position={[0, hubH / 2, 0]}>
        <sphereGeometry args={[hubR * 1.28, 20, 14, 0, Math.PI * 2, 0, Math.PI / 2]} />
        <primitive object={mat} attach="material" />
      </mesh>
      {/* Hub window ring */}
      <mesh position={[0, hubH * 0.15, 0]}>
        <torusGeometry args={[hubR + 0.05, 0.28, 4, 36]} />
        <primitive object={win} attach="material" />
      </mesh>
      {/* Hub bottom docking collar */}
      <mesh position={[0, -hubH / 2 - 1.8, 0]}>
        <cylinderGeometry args={[hubR * 0.42, hubR * 0.58, 3.6, 14]} />
        <primitive object={dock} attach="material" />
      </mesh>
      <mesh position={[0, -hubH / 2 - 3.7, 0]}>
        <torusGeometry args={[hubR * 0.42, 0.22, 6, 16]} />
        <primitive object={dock} attach="material" />
      </mesh>

      {/* ── Swept pylons ── */}
      {pylons.map(({ angle, sweepY }, i) => (
        <Pylon key={i} angle={angle} hubR={hubR} ringR={innerR} sweepY={sweepY} />
      ))}
    </group>
  );
}

// ── Cylinder station ───────────────────────────────────────────────────

function CylinderExterior({ r, h }: { r: number; h: number }) {
  const mat  = useMemo(mkHull, []);
  const acc  = useMemo(mkAccent, []);
  const win  = useMemo(mkWindow, []);
  const sol  = useMemo(mkSolar, []);
  const dock = useMemo(mkDock, []);

  const ringCount  = 6;
  const solarW     = r * 0.85;
  const solarH     = r * 0.50;
  const solarAxes  = [0, Math.PI / 2, Math.PI, (Math.PI * 3) / 2];

  return (
    <group>
      {/* Main hull */}
      <mesh>
        <cylinderGeometry args={[r, r, h, 40, 1, false]} />
        <primitive object={mat} attach="material" />
      </mesh>
      {/* Window strip (slightly proud of hull) */}
      <mesh>
        <cylinderGeometry args={[r + 0.12, r + 0.12, h * 0.82, 72, 1, true]} />
        <primitive object={win} attach="material" />
      </mesh>

      {/* Structural rings */}
      {Array.from({ length: ringCount }, (_, i) => {
        const y = -h / 2 + (h / (ringCount + 1)) * (i + 1);
        return (
          <mesh key={i} position={[0, y, 0]}>
            <torusGeometry args={[r + 0.7, 0.85, 8, 56]} />
            <primitive object={acc} attach="material" />
          </mesh>
        );
      })}

      {/* Fore taper cap */}
      <mesh position={[0, h / 2, 0]}>
        <cylinderGeometry args={[r * 0.22, r, r * 0.48, 28]} />
        <primitive object={acc} attach="material" />
      </mesh>
      {/* Fore docking collar */}
      <mesh position={[0, h / 2 + r * 0.48 + 2.2, 0]}>
        <cylinderGeometry args={[r * 0.10, r * 0.17, 4.4, 14]} />
        <primitive object={dock} attach="material" />
      </mesh>
      <mesh position={[0, h / 2 + r * 0.48 + 4.5, 0]}>
        <torusGeometry args={[r * 0.13, 0.25, 6, 16]} />
        <primitive object={dock} attach="material" />
      </mesh>

      {/* Aft taper cap */}
      <mesh position={[0, -h / 2, 0]}>
        <cylinderGeometry args={[r, r * 0.32, r * 0.48, 28]} />
        <primitive object={acc} attach="material" />
      </mesh>
      {/* Engine ring */}
      <mesh position={[0, -h / 2 - r * 0.52, 0]}>
        <torusGeometry args={[r * 0.28, r * 0.09, 10, 24]} />
        <primitive object={acc} attach="material" />
      </mesh>

      {/* Solar panel wings */}
      {solarAxes.map((angle, i) => (
        <group
          key={i}
          position={[Math.cos(angle) * (r + solarW / 2 + 0.8), 0, Math.sin(angle) * (r + solarW / 2 + 0.8)]}
          rotation={[0, -angle, 0]}
        >
          <mesh>
            <boxGeometry args={[solarW, solarH, 0.20]} />
            <primitive object={sol} attach="material" />
          </mesh>
          {/* Panel divider lines */}
          {[-0.25, 0, 0.25].map((yo, j) => (
            <mesh key={j} position={[0, yo * solarH, 0]}>
              <boxGeometry args={[solarW, 0.14, 0.22]} />
              <primitive object={acc} attach="material" />
            </mesh>
          ))}
        </group>
      ))}
    </group>
  );
}

// ── Sphere station ─────────────────────────────────────────────────────

function SphereExterior({ r }: { r: number }) {
  const mat  = useMemo(mkHull, []);
  const acc  = useMemo(mkAccent, []);
  const win  = useMemo(mkWindow, []);
  const dock = useMemo(mkDock, []);

  return (
    <group>
      {/* Main sphere */}
      <mesh>
        <sphereGeometry args={[r, 40, 30]} />
        <primitive object={mat} attach="material" />
      </mesh>

      {/* Latitude structural rings */}
      {([-42, -14, 14, 42] as const).map((lat, i) => {
        const rad   = lat * (Math.PI / 180);
        const y     = r * Math.sin(rad);
        const ringR = r * Math.cos(rad);
        return (
          <mesh key={i} position={[0, y, 0]}>
            <torusGeometry args={[ringR, r * 0.025, 8, 60]} />
            <primitive object={i === 1 || i === 2 ? win : acc} attach="material" />
          </mesh>
        );
      })}

      {/* Equatorial docking ports — 8 evenly spaced */}
      {Array.from({ length: 8 }, (_, i) => {
        const a = (i / 8) * Math.PI * 2;
        return (
          <group key={i}
            position={[r * 1.01 * Math.cos(a), 0, r * 1.01 * Math.sin(a)]}
            rotation={[0, -a, Math.PI / 2]}
          >
            <mesh>
              <cylinderGeometry args={[r * 0.042, r * 0.058, r * 0.14, 10]} />
              <primitive object={dock} attach="material" />
            </mesh>
          </group>
        );
      })}

      {/* North spire + docking collar */}
      <mesh position={[0, r + r * 0.09, 0]}>
        <cylinderGeometry args={[r * 0.048, r * 0.115, r * 0.36, 16]} />
        <primitive object={acc} attach="material" />
      </mesh>
      <mesh position={[0, r + r * 0.09 + r * 0.19, 0]}>
        <sphereGeometry args={[r * 0.095, 14, 10]} />
        <primitive object={mat} attach="material" />
      </mesh>

      {/* South engine cluster */}
      <mesh position={[0, -r - r * 0.07, 0]}>
        <cylinderGeometry args={[r * 0.13, r * 0.058, r * 0.30, 16]} />
        <primitive object={acc} attach="material" />
      </mesh>
      {Array.from({ length: 4 }, (_, i) => {
        const a = (i / 4) * Math.PI * 2;
        return (
          <mesh key={i} position={[r * 0.072 * Math.cos(a), -r - r * 0.22, r * 0.072 * Math.sin(a)]}>
            <cylinderGeometry args={[r * 0.024, r * 0.036, r * 0.16, 8]} />
            <primitive object={acc} attach="material" />
          </mesh>
        );
      })}
    </group>
  );
}

// ── Box station (per-module hull panels, existing approach) ────────────

const BOX_WINDOW_TYPES = new Set<ModuleType>(["habitat", "lab", "command", "medical"]);
const BOX_FIN_TYPES    = new Set<ModuleType>(["reactor", "engineering"]);
const EXTERIOR_H = 5.5;
const MARGIN     = 0.7;

function HullPanel({ module: m }: { module: StationModule }) {
  const yBase = m.floor * FLOOR_GAP;
  const w = m.rect.w + MARGIN * 2;
  const h = EXTERIOR_H;
  const d = m.rect.h + MARGIN * 2;
  const cx = m.rect.x + m.rect.w / 2;
  const cz = m.rect.y + m.rect.h / 2;

  const hullCol = useMemo(() => {
    const col = new THREE.Color("#8e9daf").lerp(new THREE.Color(MODULE_COLORS[m.type]), 0.07);
    if (m.damaged) col.lerp(new THREE.Color("#2a1a18"), 0.45);
    return col;
  }, [m.type, m.damaged]);

  const hullMat  = useMemo(() => new THREE.MeshStandardMaterial({
    color: hullCol, metalness: 0.62, roughness: 0.38,
    ...(m.damaged ? { transparent: true, opacity: 0.72 } : {}),
  }), [hullCol, m.damaged]);
  const trimMat  = useMemo(() => new THREE.MeshStandardMaterial({
    color: new THREE.Color(hullCol).lerp(new THREE.Color("#000"), 0.25), metalness: 0.7, roughness: 0.3,
  }), [hullCol]);
  const winMat   = useMemo(mkWindow, []);
  const dockMat  = useMemo(mkDock, []);
  const finMat2  = useMemo(mkFin, []);

  const windowCount = BOX_WINDOW_TYPES.has(m.type) ? Math.max(1, Math.floor(w / 5)) : 0;

  return (
    <group position={[cx, yBase + h / 2, cz]}>
      <mesh><boxGeometry args={[w, h, d]} /><primitive object={hullMat} attach="material" /></mesh>
      <mesh position={[0,  h / 2 - 0.15, 0]}><boxGeometry args={[w + 0.05, 0.30, d + 0.05]} /><primitive object={trimMat} attach="material" /></mesh>
      <mesh position={[0, -h / 2 + 0.15, 0]}><boxGeometry args={[w + 0.05, 0.30, d + 0.05]} /><primitive object={trimMat} attach="material" /></mesh>

      {windowCount > 0 && (() => {
        const useX   = w >= d;
        const span   = useX ? w : d;
        const step   = span / (windowCount + 1);
        return Array.from({ length: windowCount }, (_, i) => {
          const off = -span / 2 + step * (i + 1);
          return (
            <group key={i}>
              <mesh position={useX ? [off, 0, -d / 2 - 0.06] : [-w / 2 - 0.06, 0, off]}>
                <boxGeometry args={[1.1, 0.75, 0.12]} /><primitive object={winMat} attach="material" />
              </mesh>
              <mesh position={useX ? [off, 0, d / 2 + 0.06] : [w / 2 + 0.06, 0, off]}>
                <boxGeometry args={[1.1, 0.75, 0.12]} /><primitive object={winMat} attach="material" />
              </mesh>
            </group>
          );
        });
      })()}

      {m.hullAirlock && m.type === "airlock" && (
        <group position={[0, -h * 0.1, -d / 2 - 1.0]}>
          <mesh><cylinderGeometry args={[1.1, 1.35, 1.8, 14]} /><primitive object={dockMat} attach="material" /></mesh>
          <mesh position={[0, 0.95, 0]}><torusGeometry args={[1.0, 0.15, 8, 16]} /><primitive object={dockMat} attach="material" /></mesh>
        </group>
      )}

      {BOX_FIN_TYPES.has(m.type) && [0.3, 0, -0.3].map((yo, k) => (
        <group key={k}>
          <mesh position={[0, yo * h, d / 2 + 2.0]}><boxGeometry args={[w * 0.75, 2.6, 0.14]} /><primitive object={finMat2} attach="material" /></mesh>
          <mesh position={[0, yo * h, -d / 2 - 2.0]}><boxGeometry args={[w * 0.75, 2.6, 0.14]} /><primitive object={finMat2} attach="material" /></mesh>
        </group>
      ))}
    </group>
  );
}

// ── Main export ────────────────────────────────────────────────────────

export function HullSkin({ layout }: { layout: StationLayout }) {
  const shape      = layout.params.stationShape ?? "box";
  const cx         = layout.bounds.w / 2;
  const cz         = layout.bounds.h / 2;
  const floorCount = layout.params.floorCount ?? 1;
  const totalH     = floorCount * FLOOR_GAP;
  const maxSpan    = Math.max(layout.bounds.w, layout.bounds.h);

  if (shape === "box") {
    return <group>{layout.modules.map(m => <HullPanel key={m.id} module={m} />)}</group>;
  }

  return (
    <group position={[cx, totalH / 2, cz]}>
      {shape === "ring"     && <RingExterior     r={maxSpan * 0.42} />}
      {shape === "cylinder" && <CylinderExterior r={maxSpan * 0.26} h={Math.max(totalH * 1.9, maxSpan * 0.56)} />}
      {shape === "sphere"   && <SphereExterior   r={maxSpan * 0.38} />}
    </group>
  );
}
