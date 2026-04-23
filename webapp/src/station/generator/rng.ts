import seedrandom from "seedrandom";

export type Rng = ReturnType<typeof makeRng>;

export function makeRng(seed: string) {
  const prng = seedrandom(seed);
  return {
    float: () => prng(),
    int: (min: number, max: number) => Math.floor(prng() * (max - min + 1)) + min,
    pick: <T>(arr: T[]): T => arr[Math.floor(prng() * arr.length)],
    shuffle: <T>(arr: T[]): T[] => {
      const out = [...arr];
      for (let i = out.length - 1; i > 0; i--) {
        const j = Math.floor(prng() * (i + 1));
        [out[i], out[j]] = [out[j], out[i]];
      }
      return out;
    },
    bool: (chance = 0.5) => prng() < chance,
  };
}
