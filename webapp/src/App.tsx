import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import type { MLCEngineInterface } from "@mlc-ai/web-llm";
import "./App.css";

type NavigatorWithGpu = Navigator & {
  gpu?: {
    requestAdapter: () => Promise<{
      limits: { maxComputeWorkgroupStorageSize: number };
    } | null>;
  };
};

type RowRecord = Record<string, string>;

type CharacterSheetSection = {
  name: string;
  headers: string[];
  rows: RowRecord[];
};

type CharacterSheet = {
  sections: CharacterSheetSection[];
};

type CharacterOutputEffect = {
  id: string;
  kind: "add_row" | "adjust_field" | "set_field" | "remove_row" | "note";
  section: string;
  matchField?: string;
  matchValue?: string;
  field?: string;
  delta?: number;
  value?: string;
  row?: RowRecord;
  summary: string;
};

type GeneratedNarrativeOutput = {
  narrative: string;
  importance: number;
  effects: CharacterOutputEffect[];
};

type IosLiteGeneratorOutput = Array<{ generated_text?: string }>;
type IosLiteGenerator = (
  input: string,
  options?: {
    max_new_tokens?: number;
    temperature?: number;
    do_sample?: boolean;
  },
) => Promise<IosLiteGeneratorOutput>;

type TraitSignal = {
  section: string;
  label: string;
  value: number | null;
  emphasis: number | null;
  direction: "direct" | "inverse" | "neutral" | "unspecified";
  intensity: number;
  relevance: number;
  priority: number;
  reasons: string[];
};

type TraitDecisionProfile = {
  cleanSituation: string;
  tags: string[];
  topSignals: TraitSignal[];
  focusSignals: TraitSignal[];
  profileText: string;
};

type DeterministicActionPlan = {
  characterName: string;
  objective: string;
  primaryBehavior: string;
  secondaryBehavior: string;
  firstAction: string;
  secondAction: string;
  consequence: string;
  anchorTokens: string[];
  importance: number;
};

type WebLlmPromptBundle = {
  systemPrompt: string;
  userPrompt: string;
};

type PersonaPrimeResult = {
  personaLock: string;
  primeSystemPrompt: string;
  primeUserPrompt: string;
  rawModelOutput: string;
  source: "webllm" | "ios-lite" | "fallback";
};

type DebugPromptSnapshot = {
  activePath: "webllm" | "ios-lite" | "heuristic-fallback";
  cleanSituation: string;
  tags: string[];
  profileText: string;
  actionPlan: string;
  personaSeed: string;
  personaPrimeSystemPrompt: string;
  personaPrimeUserPrompt: string;
  personaLock: string;
  personaPrimeOutput: string;
  personaPrimeSource: "webllm" | "ios-lite" | "fallback";
  webLlmSystemPrompt: string;
  webLlmUserPrompt: string;
  iosLitePrompt: string;
};

const WEBLLM_MODELS = [
  {
    id: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
    label: "Llama 3.2 1B q4f16 (recommended mobile)",
  },
  {
    id: "Llama-3.2-1B-Instruct-q4f32_1-MLC",
    label: "Llama 3.2 1B q4f32 (higher quality)",
  },
  {
    id: "Llama-3.2-3B-Instruct-q4f16_1-MLC",
    label: "Llama 3.2 3B q4f16 (slower on mobile)",
  },
];

const REQUIRED_WORKGROUP_STORAGE_SIZE = 32768;
const DEFAULT_SITUATION =
  "A rival guild demands tribute from #Lila# while reaching for *blaster* on the counter.";

const STOP_WORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "by",
  "for",
  "from",
  "has",
  "he",
  "in",
  "is",
  "it",
  "its",
  "of",
  "on",
  "or",
  "that",
  "the",
  "to",
  "was",
  "were",
  "will",
  "with",
]);

const SITUATION_TAG_KEYWORDS: Array<{ tag: string; keywords: string[] }> = [
  {
    tag: "danger",
    keywords: ["threat", "attack", "ambush", "fight", "kill", "raid", "hostile"],
  },
  {
    tag: "social",
    keywords: ["talk", "speak", "argue", "convince", "negotiate", "ask", "demand"],
  },
  {
    tag: "stealth",
    keywords: ["stealth", "sneak", "hidden", "quiet", "undetected", "secret"],
  },
  {
    tag: "technology",
    keywords: ["computer", "terminal", "hack", "machine", "engine", "device", "drone"],
  },
  {
    tag: "authority",
    keywords: ["order", "command", "law", "rule", "captain", "officer", "superior"],
  },
  {
    tag: "resource",
    keywords: ["supply", "cargo", "loot", "trade", "payment", "tribute", "item"],
  },
];

const TRAIT_HINTS_BY_TAG: Record<string, string[]> = {
  danger: [
    "violence",
    "anger",
    "fear",
    "anxiety",
    "trauma",
    "vitality",
    "strength",
    "agility",
    "pain tolerance",
    "personal safety",
    "team safety",
  ],
  social: [
    "trust",
    "sociability",
    "diplomacy",
    "shyness",
    "honesty",
    "secrecy",
    "leadership",
    "loyalty",
    "arrogance",
    "confidence",
  ],
  stealth: [
    "secrecy",
    "attention span",
    "patience",
    "goal oriented",
    "rule-following",
    "anxiety",
    "doubt",
  ],
  technology: [
    "good with computers",
    "good with machines",
    "problem solving",
    "intelligence",
    "perception",
  ],
  authority: [
    "leadership",
    "rule-following",
    "rebellious",
    "honor",
    "pride",
    "loyalty",
  ],
  resource: [
    "wealth",
    "money management",
    "indulgence",
    "immediate gratification",
    "organization",
    "good with hands",
  ],
};

const normalizeName = (value: string) => value.trim().toLowerCase();

const clampZeroToTen = (value: number) =>
  Math.max(0, Math.min(10, Math.round(value)));

const detectIOS = (): boolean => {
  if (typeof navigator === "undefined") {
    return false;
  }

  const userAgent = navigator.userAgent ?? "";
  const iOSInUserAgent = /iPad|iPhone|iPod/i.test(userAgent);
  const iPadDesktopMode =
    navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1;
  return iOSInUserAgent || iPadDesktopMode;
};

const coerceContent = (content: unknown): string => {
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") {
          return part;
        }
        if (
          part &&
          typeof part === "object" &&
          "text" in part &&
          typeof (part as { text?: unknown }).text === "string"
        ) {
          return (part as { text: string }).text;
        }
        return "";
      })
      .join("");
  }

  return "";
};

const parseTaggedValues = (text: string, regex: RegExp): string[] => {
  const seen = new Set<string>();
  const values: string[] = [];
  for (const match of text.matchAll(regex)) {
    const value = (match[1] ?? "").trim();
    if (!value) {
      continue;
    }
    const key = normalizeName(value);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    values.push(value);
  }
  return values;
};

const findFieldByKeywords = (
  headers: string[],
  keywords: string[],
): string | undefined =>
  headers.find((header) =>
    keywords.some((keyword) => normalizeName(header).includes(keyword)),
  );

const findSection = (
  sheet: CharacterSheet,
  keywords: string[],
): CharacterSheetSection | undefined =>
  sheet.sections.find((section) =>
    keywords.some((keyword) => normalizeName(section.name).includes(keyword)),
  );

const findIdentifyingSection = (
  sheet: CharacterSheet,
): CharacterSheetSection | undefined =>
  sheet.sections.find((section) => {
    const normalized = normalizeName(section.name);
    return (
      normalized === "identifying traits" ||
      normalized.includes("identifying")
    );
  });

const getIdentifyingFields = (section: CharacterSheetSection) => {
  const traitField =
    findFieldByKeywords(section.headers, ["trait"]) ?? section.headers[0];
  const valueField =
    findFieldByKeywords(section.headers, ["value", "rating"]) ??
    section.headers[1];
  const emphasisField = findFieldByKeywords(section.headers, [
    "emphasis",
    "weight",
    "importance",
  ]);
  return { traitField, valueField, emphasisField };
};

const cloneSheet = (sheet: CharacterSheet): CharacterSheet =>
  JSON.parse(JSON.stringify(sheet)) as CharacterSheet;

const ensureSection = (
  sheet: CharacterSheet,
  sectionName: string,
  defaultHeaders: string[],
): CharacterSheetSection => {
  const existing = findSection(sheet, [normalizeName(sectionName)]);
  if (existing) {
    return existing;
  }

  const section: CharacterSheetSection = {
    name: sectionName,
    headers: [...defaultHeaders],
    rows: [],
  };
  sheet.sections.push(section);
  return section;
};

const ensureHeader = (section: CharacterSheetSection, header: string) => {
  if (section.headers.includes(header)) {
    return;
  }
  section.headers.push(header);
  section.rows.forEach((row) => {
    row[header] = row[header] ?? "";
  });
};

const parseCharacterSheetCsv = (csvText: string): CharacterSheet => {
  const parsed = Papa.parse<string[]>(csvText, {
    header: false,
    skipEmptyLines: false,
  });
  if (parsed.errors.length > 0) {
    throw new Error(parsed.errors[0]?.message ?? "Invalid CSV.");
  }

  const sections: CharacterSheetSection[] = [];
  let currentSection: CharacterSheetSection | null = null;

  const ensureCurrentSection = (fallbackName: string) => {
    if (!currentSection) {
      currentSection = { name: fallbackName, headers: [], rows: [] };
      sections.push(currentSection);
    }
    return currentSection;
  };

  for (const rawRow of parsed.data) {
    const row = rawRow.map((cell) => `${cell ?? ""}`.trim());
    if (row.every((cell) => !cell)) {
      continue;
    }

    const firstNonEmptyIndex = row.findIndex((cell) => cell.length > 0);
    if (firstNonEmptyIndex === -1) {
      continue;
    }
    const firstCell = row[firstNonEmptyIndex];

    if (firstCell.startsWith("!!")) {
      const sectionName =
        firstCell.replace(/^!!\s*/, "").trim() || "Unnamed Section";
      currentSection = { name: sectionName, headers: [], rows: [] };
      sections.push(currentSection);
      continue;
    }

    if (row.some((cell) => cell.startsWith("!"))) {
      const section = ensureCurrentSection("General");
      section.headers = row.map((cell, index) => {
        const clean = cell.replace(/^!\s*/, "").trim();
        return clean || `field_${index + 1}`;
      });
      continue;
    }

    const section = ensureCurrentSection("General");
    if (section.headers.length === 0) {
      section.headers = row.map((_, index) => `field_${index + 1}`);
    }
    while (row.length > section.headers.length) {
      section.headers.push(`field_${section.headers.length + 1}`);
    }

    const record: RowRecord = {};
    section.headers.forEach((header, index) => {
      record[header] = row[index] ?? "";
    });
    if (Object.values(record).some((value) => value.trim() !== "")) {
      section.rows.push(record);
    }
  }

  if (sections.length === 0) {
    throw new Error("No sections found. Expected rows starting with !! and !.");
  }
  return { sections };
};

const serializeCharacterSheetCsv = (sheet: CharacterSheet): string => {
  const rows: string[][] = [];
  sheet.sections.forEach((section) => {
    rows.push([`!!${section.name}`]);
    rows.push(section.headers.map((header) => `!${header}`));
    section.rows.forEach((row) => {
      rows.push(section.headers.map((header) => row[header] ?? ""));
    });
    rows.push([]);
  });

  return Papa.unparse(rows);
};

const buildIdentifyingPreview = (sheet: CharacterSheet): string[] => {
  const identifyingSection = findIdentifyingSection(sheet);

  if (!identifyingSection) {
    return [];
  }

  const { traitField, valueField, emphasisField } =
    getIdentifyingFields(identifyingSection);
  const traitRows = identifyingSection.rows.filter(
    (row) => (row[traitField] ?? "").trim() !== "",
  );

  return traitRows.slice(0, 5).map((row) => {
    const trait = row[traitField] || traitField;
    const value = valueField ? row[valueField] : "";
    const emphasis = emphasisField ? row[emphasisField] : "";
    if (emphasis) {
      return `${trait}: ${value} (emphasis ${emphasis})`;
    }
    return value ? `${trait}: ${value}` : trait;
  });
};

const sectionContainsEntity = (
  section: CharacterSheetSection | undefined,
  entity: string,
): boolean => {
  if (!section) {
    return false;
  }

  const entityName = normalizeName(entity);
  const nameField =
    findFieldByKeywords(section.headers, [
      "name",
      "person",
      "connection",
      "item",
      "object",
      "trait",
      "goal",
    ]) ?? section.headers[0];

  return section.rows.some((row) => normalizeName(row[nameField] ?? "") === entityName);
};

const createRowForEntity = (
  section: CharacterSheetSection | undefined,
  entity: string,
): RowRecord => {
  const headers =
    section?.headers.length && section.headers.length > 0
      ? [...section.headers]
      : ["name", "value", "emphasis", "notes"];
  const row: RowRecord = {};
  headers.forEach((header) => {
    row[header] = "";
  });

  const nameField =
    findFieldByKeywords(headers, [
      "name",
      "person",
      "connection",
      "item",
      "object",
      "trait",
    ]) ?? headers[0];
  row[nameField] = entity;

  const quantityField = findFieldByKeywords(headers, ["quantity", "count"]);
  if (quantityField) {
    row[quantityField] = "1";
  }

  const valueField = findFieldByKeywords(headers, ["value"]);
  if (valueField && !row[valueField]) {
    row[valueField] = "5";
  }

  const emphasisField = findFieldByKeywords(headers, ["emphasis", "importance"]);
  if (emphasisField && !row[emphasisField]) {
    row[emphasisField] = "5";
  }

  const noteField = findFieldByKeywords(headers, ["note", "detail", "summary"]);
  if (noteField) {
    row[noteField] = "Added from prompt tag.";
  }

  return row;
};

const buildTagEffects = (
  sheet: CharacterSheet,
  people: string[],
  items: string[],
): CharacterOutputEffect[] => {
  const effects: CharacterOutputEffect[] = [];
  const connectionsSection = findSection(sheet, ["connection"]);
  const inventorySection = findSection(sheet, ["inventory", "item"]);

  people.forEach((person, index) => {
    if (!sectionContainsEntity(connectionsSection, person)) {
      effects.push({
        id: `tag-person-${index}-${person}`,
        kind: "add_row",
        section: connectionsSection?.name ?? "Connections",
        row: createRowForEntity(connectionsSection, person),
        summary: `Add connection entry for ${person}.`,
      });
    }
  });

  items.forEach((item, index) => {
    if (!sectionContainsEntity(inventorySection, item)) {
      effects.push({
        id: `tag-item-${index}-${item}`,
        kind: "add_row",
        section: inventorySection?.name ?? "Inventory",
        row: createRowForEntity(inventorySection, item),
        summary: `Add inventory entry for ${item}.`,
      });
    }
  });

  return effects;
};

const parseImportance = (value: unknown): number => {
  const numeric = Number.parseFloat(String(value ?? ""));
  if (!Number.isFinite(numeric)) {
    return 5;
  }
  return clampZeroToTen(numeric);
};

const parseStructuredOutput = (
  text: string,
): Omit<GeneratedNarrativeOutput, "effects"> & {
  effects: Array<Partial<CharacterOutputEffect>>;
} | null => {
  const cleaned = text
    .replace(/```json/gi, "")
    .replace(/```/g, "")
    .trim();
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) {
    return null;
  }

  try {
    const parsed = JSON.parse(cleaned.slice(start, end + 1)) as {
      narrative?: unknown;
      importance?: unknown;
      effects?: unknown;
    };
    if (typeof parsed.narrative !== "string") {
      return null;
    }

    const effects = Array.isArray(parsed.effects)
      ? (parsed.effects as Array<Partial<CharacterOutputEffect>>)
      : [];
    return {
      narrative: parsed.narrative.trim(),
      importance: parseImportance(parsed.importance),
      effects,
    };
  } catch {
    return null;
  }
};

const parseOptionalNumeric = (value: string | undefined): number | null => {
  const numeric = Number.parseFloat(String(value ?? ""));
  return Number.isFinite(numeric) ? clampZeroToTen(numeric) : null;
};

const cleanSituationText = (prompt: string): string =>
  prompt
    .replace(/#([^#\n]+?)#/g, "$1")
    .replace(/\*([^*\n]+?)\*/g, "$1")
    .trim();

const tokenizeText = (text: string): string[] =>
  normalizeName(text)
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .map((token) => token.trim())
    .filter((token) => token.length > 1 && !STOP_WORDS.has(token));

const buildSituationTags = (
  cleanSituation: string,
  people: string[],
  items: string[],
): string[] => {
  const lower = normalizeName(cleanSituation);
  const tags = new Set<string>();

  SITUATION_TAG_KEYWORDS.forEach(({ tag, keywords }) => {
    if (keywords.some((keyword) => lower.includes(keyword))) {
      tags.add(tag);
    }
  });

  if (people.length > 0) {
    tags.add("social");
  }
  if (items.length > 0) {
    tags.add("resource");
  }

  return [...tags];
};

const extractTraitSignals = (sheet: CharacterSheet): TraitSignal[] => {
  const signals: TraitSignal[] = [];

  sheet.sections.forEach((section) => {
    if (section.rows.length === 0 || section.headers.length === 0) {
      return;
    }

    const labelField =
      findFieldByKeywords(section.headers, [
        "trait",
        "motivation",
        "goal",
        "person",
        "item",
        "name",
      ]) ?? section.headers[0];
    const valueField = findFieldByKeywords(section.headers, ["value", "rating"]);
    const emphasisField = findFieldByKeywords(section.headers, [
      "emphasis",
      "priority",
      "importance",
      "strength",
      "weight",
    ]);

    section.rows.forEach((row) => {
      const label = (row[labelField] ?? "").trim();
      if (!label) {
        return;
      }

      const value = valueField ? parseOptionalNumeric(row[valueField]) : null;
      const emphasis = emphasisField
        ? parseOptionalNumeric(row[emphasisField])
        : null;

      if (value === null && emphasis === null) {
        return;
      }

      const intensity = value === null ? 2.5 : Math.abs(value - 5);
      const direction: TraitSignal["direction"] =
        value === null
          ? "unspecified"
          : value > 5
            ? "direct"
            : value < 5
              ? "inverse"
              : "neutral";

      signals.push({
        section: section.name,
        label,
        value,
        emphasis,
        direction,
        intensity,
        relevance: 0,
        priority: 0,
        reasons: [],
      });
    });
  });

  return signals;
};

const scoreTraitSignal = (
  signal: TraitSignal,
  situationTokens: Set<string>,
  situationTags: string[],
  people: string[],
  items: string[],
): TraitSignal => {
  const reasons: string[] = [];
  let relevance = 0;
  const labelNormalized = normalizeName(signal.label);
  const labelTokens = tokenizeText(signal.label);
  const overlapTokens = labelTokens.filter((token) => situationTokens.has(token));

  if (overlapTokens.length > 0) {
    relevance += 3 + overlapTokens.length * 0.8;
    reasons.push(`keyword overlap: ${overlapTokens.join(", ")}`);
  }

  situationTags.forEach((tag) => {
    const hints = TRAIT_HINTS_BY_TAG[tag] ?? [];
    if (hints.some((hint) => labelNormalized.includes(hint))) {
      relevance += 2.4;
      reasons.push(`matches ${tag} tag`);
    }
  });

  if (
    people.length > 0 &&
    ["trust", "sociability", "diplomacy", "shyness", "honesty", "secrecy"].some(
      (keyword) => labelNormalized.includes(keyword),
    )
  ) {
    relevance += 1.6;
    reasons.push("boosted for named people");
  }

  if (
    items.length > 0 &&
    ["computers", "machines", "hands", "resource", "wealth"].some((keyword) =>
      labelNormalized.includes(keyword),
    )
  ) {
    relevance += 1.3;
    reasons.push("boosted for named items");
  }

  if (relevance === 0 && (signal.emphasis ?? 0) >= 8) {
    relevance = 0.7;
    reasons.push("high emphasis fallback");
  }

  const emphasisScore = signal.emphasis ?? 5;
  const emphasisFactor = 0.55 + emphasisScore / 20;
  const intensityFactor = 0.4 + signal.intensity / 10;
  const priority = relevance * emphasisFactor * intensityFactor;

  return {
    ...signal,
    relevance,
    priority,
    reasons,
  };
};

const describeTraitBehavior = (signal: TraitSignal): string => {
  const intensityWord =
    signal.intensity >= 4
      ? "very strongly"
      : signal.intensity >= 2.5
        ? "strongly"
        : signal.intensity >= 1.2
          ? "moderately"
          : "slightly";

  if (signal.direction === "inverse") {
    return `${intensityWord} inverse expression of ${signal.label} (low ${signal.label})`;
  }
  if (signal.direction === "neutral") {
    return `balanced expression of ${signal.label}`;
  }
  if (signal.direction === "unspecified") {
    return `high motivational pressure around ${signal.label}`;
  }
  return `${intensityWord} direct expression of ${signal.label}`;
};

const inferActionCueFromSignal = (signal: TraitSignal): string => {
  const label = normalizeName(signal.label);
  const inverse = signal.direction === "inverse";
  if (label.includes("shy") || label.includes("sociab")) {
    return inverse ? "directly confronting others" : "careful social restraint";
  }
  if (label.includes("diplom") || label.includes("trust") || label.includes("honesty")) {
    return inverse ? "hard bargaining and guarded wording" : "measured diplomacy";
  }
  if (label.includes("violence") || label.includes("anger") || label.includes("fear")) {
    return inverse ? "controlled de-escalation" : "readiness for confrontation";
  }
  if (label.includes("secrecy") || label.includes("rule")) {
    return inverse ? "open, blunt communication" : "tight information control";
  }
  if (label.includes("goal") || label.includes("organization") || label.includes("leadership")) {
    return inverse ? "improvised reaction" : "structured command decisions";
  }
  return inverse ? "a contrarian tactical move" : "a decisive, practiced move";
};

const buildTraitDecisionProfile = (
  sheet: CharacterSheet,
  situationPrompt: string,
  people: string[],
  items: string[],
): TraitDecisionProfile => {
  const cleanSituation = cleanSituationText(situationPrompt);
  const situationTokens = new Set(tokenizeText(cleanSituation));
  const tags = buildSituationTags(cleanSituation, people, items);
  const scoredSignals = extractTraitSignals(sheet)
    .map((signal) => scoreTraitSignal(signal, situationTokens, tags, people, items))
    .sort((a, b) => {
      if (b.priority !== a.priority) {
        return b.priority - a.priority;
      }
      const emphasisDelta = (b.emphasis ?? 0) - (a.emphasis ?? 0);
      if (emphasisDelta !== 0) {
        return emphasisDelta;
      }
      const intensityDelta = b.intensity - a.intensity;
      if (intensityDelta !== 0) {
        return intensityDelta;
      }
      return a.label.localeCompare(b.label);
    });

  const relevant = scoredSignals.filter((signal) => signal.relevance > 0);
  const topSignals = (relevant.length > 0 ? relevant : scoredSignals).slice(0, 10);
  const focusSignals = [...(relevant.length > 0 ? relevant : scoredSignals)]
    .sort((a, b) => {
      const emphasisDelta = (b.emphasis ?? 5) - (a.emphasis ?? 5);
      if (emphasisDelta !== 0) {
        return emphasisDelta;
      }
      if (b.priority !== a.priority) {
        return b.priority - a.priority;
      }
      if (b.intensity !== a.intensity) {
        return b.intensity - a.intensity;
      }
      return a.label.localeCompare(b.label);
    })
    .slice(0, 5);
  const profileLines = [
    "Trait decision profile (silent context):",
    `Situation tags: ${tags.length > 0 ? tags.join(", ") : "general"}.`,
    "Top 5 emphasized weighted traits for LLM context:",
    ...focusSignals.map((signal, index) => {
      const valueText = signal.value === null ? "n/a" : String(signal.value);
      const emphasisText = signal.emphasis === null ? "n/a" : String(signal.emphasis);
      return `${index + 1}. ${signal.label} -> ${describeTraitBehavior(signal)} | value=${valueText}, emphasis=${emphasisText}, priority=${signal.priority.toFixed(2)}.`;
    }),
    "Ranked behavior drivers:",
    ...topSignals.map((signal, index) => {
      const valueText = signal.value === null ? "n/a" : String(signal.value);
      const emphasisText = signal.emphasis === null ? "n/a" : String(signal.emphasis);
      return `${index + 1}. ${signal.label} [${signal.section}] -> ${describeTraitBehavior(signal)} | value=${valueText}, emphasis=${emphasisText}, relevance=${signal.relevance.toFixed(2)}, priority=${signal.priority.toFixed(2)}.`;
    }),
    "Decision policy: prioritize highest-emphasis relevant traits; if value < 5, express the inverse tendency.",
  ];

  return {
    cleanSituation,
    tags,
    topSignals,
    focusSignals,
    profileText: profileLines.join("\n"),
  };
};

const extractTopTraitsByEmphasis = (
  section: CharacterSheetSection | undefined,
  limit: number,
): TraitSignal[] => {
  if (!section || section.rows.length === 0) {
    return [];
  }

  const labelField =
    findFieldByKeywords(section.headers, ["trait", "motivation", "goal"]) ??
    section.headers[0];
  const valueField = findFieldByKeywords(section.headers, ["value", "rating"]);
  const emphasisField = findFieldByKeywords(section.headers, [
    "emphasis",
    "priority",
    "importance",
    "strength",
    "weight",
  ]);

  const collected = section.rows.reduce<TraitSignal[]>((accumulator, row) => {
      const label = (row[labelField] ?? "").trim();
      if (!label) {
        return accumulator;
      }
      const value = valueField ? parseOptionalNumeric(row[valueField]) : null;
      const emphasis = emphasisField ? parseOptionalNumeric(row[emphasisField]) : null;
      const intensity = value === null ? 2.5 : Math.abs(value - 5);
      const direction: TraitSignal["direction"] =
        value === null
          ? "unspecified"
          : value > 5
            ? "direct"
            : value < 5
              ? "inverse"
              : "neutral";

      accumulator.push({
        section: section.name,
        label,
        value,
        emphasis,
        direction,
        intensity,
        relevance: 1,
        priority: (emphasis ?? 5) * (0.4 + intensity / 10),
        reasons: [],
      });
      return accumulator;
    }, []);

  return collected.sort((a, b) => {
      const emphasisDelta = (b.emphasis ?? 5) - (a.emphasis ?? 5);
      if (emphasisDelta !== 0) {
        return emphasisDelta;
      }
      if (b.priority !== a.priority) {
        return b.priority - a.priority;
      }
      return a.label.localeCompare(b.label);
    })
    .slice(0, limit);
};

const buildPersonaSeedFromSheet = (
  sheet: CharacterSheet,
  traitProfile: TraitDecisionProfile,
): string => {
  const identifyingSection = findIdentifyingSection(sheet);
  const physicalSection = findSection(sheet, ["physical"]);
  const personalitySection = findSection(sheet, ["personality"]);
  const motivationsSection = findSection(sheet, ["motivation"]);
  const goalsSection = findSection(sheet, ["goal"]);

  const identifyingLines: string[] = [];
  if (identifyingSection) {
    const { traitField, valueField } = getIdentifyingFields(identifyingSection);
    identifyingSection.rows.slice(0, 12).forEach((row) => {
      const label = (row[traitField] ?? "").trim();
      const value = (valueField ? row[valueField] : "").trim();
      if (!label || !value) {
        return;
      }
      identifyingLines.push(`${label}: ${value}`);
    });
  }

  const topPhysical = extractTopTraitsByEmphasis(physicalSection, 5);
  const topPersonality = extractTopTraitsByEmphasis(personalitySection, 5);
  const topMotivations = extractTopTraitsByEmphasis(motivationsSection, 4);
  const topGoals = extractTopTraitsByEmphasis(goalsSection, 4);

  const formatSignal = (signal: TraitSignal) => {
    const valueText = signal.value === null ? "n/a" : String(signal.value);
    const emphasisText = signal.emphasis === null ? "n/a" : String(signal.emphasis);
    return `${signal.label} -> ${describeTraitBehavior(signal)} (value=${valueText}, emphasis=${emphasisText})`;
  };

  return [
    "Persona seed context:",
    "Identifying traits:",
    ...(identifyingLines.length > 0 ? identifyingLines.map((line) => `- ${line}`) : ["- Unknown identity"]),
    "Physical traits (weighted):",
    ...(topPhysical.length > 0 ? topPhysical.map((signal) => `- ${formatSignal(signal)}`) : ["- none"]),
    "Top 5 personality traits:",
    ...(topPersonality.length > 0
      ? topPersonality.map((signal) => `- ${formatSignal(signal)}`)
      : ["- none"]),
    "Motivations:",
    ...(topMotivations.length > 0
      ? topMotivations.map((signal) => `- ${formatSignal(signal)}`)
      : ["- none"]),
    "Goals:",
    ...(topGoals.length > 0 ? topGoals.map((signal) => `- ${formatSignal(signal)}`) : ["- none"]),
    "Situation tags likely encountered in play:",
    `- ${traitProfile.tags.join(", ") || "general"}`,
  ].join("\n");
};

const buildPersonaPrimePromptBundle = (personaSeed: string): WebLlmPromptBundle => {
  const systemPrompt = [
    "You are a persona compiler for an RPG NPC model.",
    "Return strict JSON only with key persona_lock.",
    "persona_lock: 4-6 sentences in third-person describing stable identity, physicality, impulses, motivations, and goal tension.",
    "Do not output bullet points in persona_lock.",
    "Do not copy raw numeric values.",
  ].join("\n");
  const userPrompt = [
    "Compile persona from this sheet-derived seed.",
    personaSeed,
  ].join("\n\n");
  return { systemPrompt, userPrompt };
};

const parsePersonaPrimeOutput = (text: string): string | null => {
  const cleaned = text
    .replace(/```json/gi, "")
    .replace(/```/g, "")
    .trim();
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  if (start !== -1 && end !== -1 && end > start) {
    try {
      const parsed = JSON.parse(cleaned.slice(start, end + 1)) as {
        persona_lock?: unknown;
        style_rules?: unknown;
      };
      const personaLock =
        typeof parsed.persona_lock === "string" ? parsed.persona_lock.trim() : "";
      if (personaLock) {
        return personaLock;
      }
    } catch {
      // Fallback to raw text handling below
    }
  }
  const trimmed = cleaned
    .replace(/style\s*rules\s*:/gi, "\n")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith("-"))
    .join(" ")
    .trim();
  return trimmed || null;
};

const buildFallbackPersonaLock = (personaSeed: string): string => {
  const lines = personaSeed.split("\n").filter(Boolean);
  const identifying = lines
    .filter((line) => line.startsWith("- "))
    .slice(0, 5)
    .map((line) => line.replace(/^- /, ""));
  return [
    "This character acts from a stable internal identity shaped by their background, body, and long-standing drives.",
    identifying.length > 0
      ? `Identity anchors: ${identifying.join("; ")}.`
      : "Identity anchors: unknown.",
    "Their behavior should reflect motivations and goals under pressure while remaining consistent across scenes.",
    "Narrative voice stays third-person, in-character, and action-oriented.",
  ].join(" ");
};

const isPersonaLockUsable = (personaLock: string): boolean => {
  const normalized = normalizeName(personaLock);
  if (normalized.length < 80) {
    return false;
  }
  const sentenceCount = personaLock
    .split(/[.!?]/)
    .map((part) => part.trim())
    .filter(Boolean).length;
  if (sentenceCount < 2 || sentenceCount > 8) {
    return false;
  }

  const tokens = normalized
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 0);
  if (tokens.length < 20) {
    return false;
  }

  const frequency = new Map<string, number>();
  tokens.forEach((token) => {
    frequency.set(token, (frequency.get(token) ?? 0) + 1);
  });
  const maxTokenFrequency = Math.max(...frequency.values());
  if (maxTokenFrequency / tokens.length > 0.2) {
    return false;
  }

  const bigrams = new Map<string, number>();
  for (let index = 0; index < tokens.length - 1; index += 1) {
    const key = `${tokens[index]} ${tokens[index + 1]}`;
    bigrams.set(key, (bigrams.get(key) ?? 0) + 1);
  }
  const maxBigramFrequency = bigrams.size > 0 ? Math.max(...bigrams.values()) : 0;
  if (maxBigramFrequency / tokens.length > 0.12) {
    return false;
  }

  if (
    normalized.includes("rpg npc model") ||
    normalized.includes("sex personality") ||
    normalized.includes("style rules")
  ) {
    return false;
  }

  return true;
};

const getCharacterNameFromSheet = (sheet: CharacterSheet): string => {
  const identifyingSection = findIdentifyingSection(sheet);
  if (!identifyingSection) {
    return "The NPC";
  }
  const { traitField, valueField } = getIdentifyingFields(identifyingSection);
  const nameRow = identifyingSection.rows.find(
    (row) => normalizeName(row[traitField] ?? "") === "name",
  );
  const fallbackRow = identifyingSection.rows[0];
  const candidate =
    (nameRow && valueField ? nameRow[valueField] : "") ||
    (fallbackRow && valueField ? fallbackRow[valueField] : "") ||
    (fallbackRow ? fallbackRow[traitField] : "");
  return candidate && candidate.trim() ? candidate.trim() : "The NPC";
};

const buildDeterministicActionPlan = (
  sheet: CharacterSheet,
  prompt: string,
  people: string[],
  items: string[],
  traitProfile: TraitDecisionProfile,
): DeterministicActionPlan => {
  const characterName = getCharacterNameFromSheet(sheet);
  const cleanSituation = cleanSituationText(prompt);
  const lower = normalizeName(cleanSituation);
  const topTraits = traitProfile.topSignals.slice(0, 3);
  const [primaryTrait, secondaryTrait] = topTraits;
  const primaryBehavior = primaryTrait
    ? inferActionCueFromSignal(primaryTrait)
    : "focused and pragmatic behavior";
  const secondaryBehavior = secondaryTrait
    ? inferActionCueFromSignal(secondaryTrait)
    : "measured follow-through";
  const situationBits = cleanSituation
    .split(/[.!?]/)
    .map((part) => part.trim())
    .filter(Boolean);
  const scene = situationBits[0] ?? "the confrontation";
  const primaryPerson = people[0] ?? "the counterpart";
  const primaryItem = items[0] ?? "the nearest leverage point";

  const hasThreat = ["threat", "attack", "ambush", "fight", "guards", "weapon"].some((term) =>
    lower.includes(term),
  );
  const hasDemand = ["demand", "tribute", "payment", "extort", "negotiate", "mercy"].some(
    (term) => lower.includes(term),
  );
  const hasStealth = ["sneak", "hidden", "quiet", "secret", "undetected"].some((term) =>
    lower.includes(term),
  );

  const objective = hasThreat
    ? `shut down the immediate threat around ${scene}`
    : hasDemand
      ? `regain control of the exchange around ${scene}`
      : hasStealth
        ? `secure an advantage before anyone notices the shift around ${scene}`
        : `take control of the next move in ${scene}`;

  const firstAction =
    people.length > 0 && items.length > 0
      ? `${characterName} steps between ${primaryPerson} and ${primaryItem}, then uses ${primaryBehavior} to seize initiative.`
      : people.length > 0
        ? `${characterName} closes on ${primaryPerson} and uses ${primaryBehavior} to set hard terms immediately.`
        : items.length > 0
          ? `${characterName} secures ${primaryItem} first, then uses ${primaryBehavior} to dictate pace.`
          : `${characterName} moves first with ${primaryBehavior}, forcing the scene onto a narrower track.`;

  const secondAction =
    people.length > 0
      ? `${characterName} presses ${primaryPerson} with ${secondaryBehavior}, demanding a concrete commitment before momentum can flip.`
      : `${characterName} layers in ${secondaryBehavior} to lock the next decision point before others can react.`;

  const consequence = hasThreat
    ? "That sequence collapses the threat window and leaves bystanders a clear lane to act."
    : hasDemand
      ? "That sequence reframes the standoff on immediate terms and strips room for evasive bargaining."
      : hasStealth
        ? "That sequence preserves initiative while keeping intentions difficult to read."
        : "That sequence creates immediate leverage and makes the follow-up action predictable for allies.";

  const topEmphasis =
    topTraits.length > 0
      ? topTraits.reduce((sum, signal) => sum + (signal.emphasis ?? 5), 0) / topTraits.length
      : 5;
  const pressureBonus = hasThreat ? 3 : hasDemand ? 2 : 1;
  const importance = clampZeroToTen(pressureBonus + topEmphasis * 0.6);

  return {
    characterName,
    objective,
    primaryBehavior,
    secondaryBehavior,
    firstAction,
    secondAction,
    consequence,
    anchorTokens: tokenizeText(cleanSituation).slice(0, 8),
    importance,
  };
};

const formatActionPlan = (plan: DeterministicActionPlan): string =>
  [
    `Character: ${plan.characterName}`,
    `Objective: ${plan.objective}`,
    `Primary behavior cue: ${plan.primaryBehavior}`,
    `Secondary behavior cue: ${plan.secondaryBehavior}`,
    `Action 1: ${plan.firstAction}`,
    `Action 2: ${plan.secondAction}`,
    `Immediate consequence: ${plan.consequence}`,
    `Anchor tokens: ${plan.anchorTokens.join(", ") || "none"}`,
  ].join("\n");

const buildActionPlanPromptAnchors = (plan: DeterministicActionPlan): string[] => [
  "Deterministic action anchors (reflect these beats; do not copy wording verbatim):",
  `- Objective: ${plan.objective}`,
  `- First action beat: ${plan.firstAction}`,
  `- Second action beat: ${plan.secondAction}`,
  `- Immediate consequence beat: ${plan.consequence}`,
  `- Must include at least one of these context tokens: ${
    plan.anchorTokens.slice(0, 4).join(", ") || "none"
  }`,
];

const buildHeuristicEffects = (
  sheet: CharacterSheet,
  prompt: string,
): CharacterOutputEffect[] => {
  const dangerKeywords = ["attack", "threat", "fight", "ambush", "betray"];
  const calmKeywords = ["gift", "help", "peace", "apology", "alliance"];
  const lowerPrompt = normalizeName(prompt);
  const identifyingSection = findIdentifyingSection(sheet);

  if (!identifyingSection || identifyingSection.rows.length === 0) {
    return [];
  }

  const { traitField, emphasisField } = getIdentifyingFields(identifyingSection);

  if (!emphasisField) {
    return [];
  }

  const firstRow = identifyingSection.rows[0];
  const traitName = firstRow[traitField] || "primary trait";

  let delta = 0;
  if (dangerKeywords.some((word) => lowerPrompt.includes(word))) {
    delta = 1;
  } else if (calmKeywords.some((word) => lowerPrompt.includes(word))) {
    delta = -1;
  }

  if (delta === 0) {
    return [];
  }

  return [
    {
      id: "heuristic-emphasis-shift",
      kind: "adjust_field",
      section: identifyingSection.name,
      matchField: traitField,
      matchValue: firstRow[traitField],
      field: emphasisField,
      delta,
      summary: `${delta > 0 ? "Increase" : "Decrease"} ${traitName} emphasis by ${Math.abs(delta)} due to recent events.`,
    },
  ];
};

const buildHeuristicNarrative = (
  actionPlan: DeterministicActionPlan,
): GeneratedNarrativeOutput => {
  return {
    narrative: [
      `${actionPlan.characterName} commits immediately to ${actionPlan.objective}.`,
      actionPlan.firstAction,
      actionPlan.secondAction,
      actionPlan.consequence,
    ].join(" "),
    importance: actionPlan.importance,
    effects: [],
  };
};

const normalizeEffects = (
  rawEffects: Array<Partial<CharacterOutputEffect>>,
): CharacterOutputEffect[] =>
  rawEffects
    .map((effect, index) => {
      const kind = effect.kind;
      const validKind =
        kind === "add_row" ||
        kind === "adjust_field" ||
        kind === "set_field" ||
        kind === "remove_row" ||
        kind === "note"
          ? kind
          : "note";

      return {
        id: effect.id ?? `model-effect-${index}`,
        kind: validKind,
        section: effect.section ?? "General",
        matchField: effect.matchField,
        matchValue: effect.matchValue,
        field: effect.field,
        delta:
          typeof effect.delta === "number"
            ? effect.delta
            : Number.parseFloat(String(effect.delta ?? "")),
        value:
          effect.value === undefined || effect.value === null
            ? undefined
            : String(effect.value),
        row: effect.row,
        summary:
          effect.summary && String(effect.summary).trim()
            ? String(effect.summary).trim()
            : "Apply contextual sheet update.",
      };
    })
    .map((effect) => ({
      ...effect,
      delta: Number.isFinite(effect.delta ?? Number.NaN) ? effect.delta : undefined,
    }));

const buildWebLlmPromptBundle = (
  promptText: string,
  personaLock: string,
  actionPlan: DeterministicActionPlan,
): WebLlmPromptBundle => {
  const actionAnchors = buildActionPlanPromptAnchors(actionPlan);
  const systemPrompt = [
    "Return strict JSON only.",
    "JSON schema:",
    '{"narrative":"string","importance":0-10,"effects":[{"kind":"add_row|adjust_field|set_field|remove_row|note","section":"string","matchField":"string","matchValue":"string","field":"string","delta":-3..3,"value":"string","row":{"field":"value"},"summary":"string"}]}',
    "Narrative must be third-person, in-character, and 3-5 sentences.",
    "Narrative must contain at least two explicit character actions and one immediate consequence.",
    "Do not include trait labels, style rules, or prompt text verbatim in narrative.",
    "Do not use abstract filler phrases such as 'core-trait behavior', 'under pressure', or 'control the exchange'.",
    "Effects should be short, concrete, and relevant.",
    ...actionAnchors,
  ].join("\n");

  const userPrompt = [
    "You are the person described below:",
    personaLock || "A composed, goal-driven operative with a stable identity under pressure.",
    "",
    "You find yourself in this situation:",
    promptText,
    "",
    "Describe what you do in 3-5 sentences in response to the situation",
  ].join("\n");

  return { systemPrompt, userPrompt };
};

const buildIosLitePromptText = (
  promptText: string,
  personaLock: string,
  actionPlan: DeterministicActionPlan,
): string => {
  const actionAnchors = buildActionPlanPromptAnchors(actionPlan);
  return [
    "Return JSON only with keys narrative, importance, effects.",
    "Narrative is 3-5 sentences, third-person in-character story prose.",
    "Narrative includes at least two explicit character actions and one immediate consequence.",
    "Importance is 0-10.",
    "Effects is an array with updates.",
    "Do not include trait labels, style rules, or prompt text verbatim in narrative.",
    "Avoid abstract filler phrases like 'core-trait behavior', 'under pressure', or 'control the exchange'.",
    ...actionAnchors,
    "You are the person described below:",
    personaLock || "No persona lock available.",
    "You find yourself in this situation:",
    promptText,
    "Describe what you do in 3-5 sentences in response to the situation",
  ].join("\n");
};

const enforceNarrativeStyle = (narrative: string, cleanSituation: string): string => {
  const sentences = narrative
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
  const seen = new Set<string>();
  const deduped = sentences.filter((sentence) => {
    const normalized = normalizeName(sentence);
    if (seen.has(normalized)) {
      return false;
    }
    seen.add(normalized);
    return true;
  });
  let cleaned = deduped.join(" ");

  if (cleanSituation && cleaned.includes(cleanSituation)) {
    cleaned = cleaned.replace(cleanSituation, "the situation");
  }

  cleaned = cleaned
    .replace(/\btrait decision profile\b/gi, "")
    .replace(/\bvalue\s*=\s*\d+(\.\d+)?\b/gi, "")
    .replace(/\bemphasis\s*=\s*\d+(\.\d+)?\b/gi, "")
    .replace(/\bpriority\s*=\s*\d+(\.\d+)?\b/gi, "")
    .replace(/\s{2,}/g, " ")
    .trim();

  return cleaned || narrative;
};

const NARRATIVE_ACTION_VERBS = [
  "grabs",
  "draws",
  "steps",
  "moves",
  "signals",
  "orders",
  "blocks",
  "takes",
  "hands",
  "shoves",
  "aims",
  "fires",
  "negotiates",
  "demands",
  "warns",
  "backs",
  "pulls",
  "pushes",
  "points",
  "nods",
  "turns",
];

const extractNarrativeFromRawModelText = (rawText: string): string | null => {
  const cleaned = rawText.replace(/```json/gi, "").replace(/```/g, "").trim();
  if (!cleaned) {
    return null;
  }

  const lines = cleaned
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .filter(
      (line) =>
        !/^(return json|json schema|importance|effects|you are the person described below|you find yourself|describe what you do)/i.test(
          line,
        ),
    );

  const candidate = lines.join(" ").replace(/\s{2,}/g, " ").trim();
  if (candidate.length < 30) {
    return null;
  }

  const sentences = candidate
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
  return sentences.slice(0, 5).join(" ");
};

const isNarrativeWeak = (
  narrative: string,
  cleanSituation: string,
  actionPlan: DeterministicActionPlan,
  people: string[],
  items: string[],
): boolean => {
  const lowerNarrative = normalizeName(narrative);
  const lowerSituation = normalizeName(cleanSituation);
  if (!lowerNarrative) {
    return true;
  }

  const bannedPhrases = [
    "core-trait behavior",
    "under pressure",
    "control the exchange",
    "trait decision profile",
  ];
  if (bannedPhrases.some((phrase) => lowerNarrative.includes(phrase))) {
    return true;
  }

  if (lowerSituation.length > 20 && lowerNarrative.includes(lowerSituation)) {
    return true;
  }

  if (
    people.length > 0 &&
    !people.some((person) => lowerNarrative.includes(normalizeName(person)))
  ) {
    return true;
  }

  if (
    items.length > 0 &&
    !items.some((item) => lowerNarrative.includes(normalizeName(item)))
  ) {
    return true;
  }

  const overlapWithPlanTokens = actionPlan.anchorTokens.filter((token) =>
    lowerNarrative.includes(token),
  ).length;
  if (actionPlan.anchorTokens.length > 0 && overlapWithPlanTokens === 0) {
    return true;
  }

  const actionHits = NARRATIVE_ACTION_VERBS.filter((verb) =>
    lowerNarrative.includes(verb),
  ).length;
  if (actionHits < 2) {
    return true;
  }

  const consequenceMarkers = [
    "forcing",
    "leaving",
    "so ",
    "therefore",
    "which",
    "before anyone",
    "as a result",
  ];
  if (!consequenceMarkers.some((marker) => lowerNarrative.includes(marker))) {
    return true;
  }

  return false;
};

const applyEffectToSheet = (sheet: CharacterSheet, effect: CharacterOutputEffect) => {
  const section = findSection(sheet, [normalizeName(effect.section)]);

  const resolveRow = (targetSection: CharacterSheetSection): RowRecord | undefined => {
    if (effect.matchField && effect.matchValue) {
      const fieldName =
        targetSection.headers.find(
          (header) => normalizeName(header) === normalizeName(effect.matchField ?? ""),
        ) ?? effect.matchField;
      return targetSection.rows.find(
        (row) => normalizeName(row[fieldName] ?? "") === normalizeName(effect.matchValue ?? ""),
      );
    }
    return targetSection.rows[0];
  };

  if (effect.kind === "add_row") {
    const targetSection = ensureSection(
      sheet,
      effect.section,
      effect.row ? Object.keys(effect.row) : ["name", "value", "emphasis", "notes"],
    );
    const row = effect.row ?? {};
    Object.keys(row).forEach((header) => ensureHeader(targetSection, header));
    const nextRow: RowRecord = {};
    targetSection.headers.forEach((header) => {
      nextRow[header] = row[header] ?? "";
    });
    targetSection.rows.push(nextRow);
    return;
  }

  if (!section) {
    return;
  }

  if (effect.kind === "remove_row") {
    const row = resolveRow(section);
    if (!row) {
      return;
    }
    const index = section.rows.indexOf(row);
    if (index >= 0) {
      section.rows.splice(index, 1);
    }
    return;
  }

  if (effect.kind === "set_field" || effect.kind === "adjust_field") {
    if (!effect.field) {
      return;
    }
    ensureHeader(section, effect.field);
    const row = resolveRow(section) ?? section.rows[0];
    if (!row) {
      return;
    }

    if (effect.kind === "set_field") {
      row[effect.field] = effect.value ?? row[effect.field] ?? "";
      return;
    }

    const current = Number.parseFloat(row[effect.field] ?? "0");
    const delta = effect.delta ?? 0;
    const nextNumeric = Number.isFinite(current)
      ? current + delta
      : Number.parseFloat(String(delta));

    if (
      normalizeName(effect.field).includes("value") ||
      normalizeName(effect.field).includes("emphasis") ||
      normalizeName(effect.field).includes("importance")
    ) {
      row[effect.field] = String(clampZeroToTen(nextNumeric));
    } else {
      row[effect.field] = String(nextNumeric);
    }
  }
};

const appendHistoryEvent = (
  sheet: CharacterSheet,
  narrative: string,
  importance: number,
  effects: CharacterOutputEffect[],
) => {
  const historySection =
    findSection(sheet, ["history"]) ??
    ensureSection(sheet, "History", ["event", "importance", "effects"]);

  const eventField =
    findFieldByKeywords(historySection.headers, ["event", "history", "narrative"]) ??
    "event";
  const importanceField =
    findFieldByKeywords(historySection.headers, ["importance", "weight"]) ??
    "importance";
  const effectsField =
    findFieldByKeywords(historySection.headers, ["effect", "change"]) ?? "effects";

  ensureHeader(historySection, eventField);
  ensureHeader(historySection, importanceField);
  ensureHeader(historySection, effectsField);

  const row: RowRecord = {};
  historySection.headers.forEach((header) => {
    row[header] = "";
  });
  row[eventField] = narrative;
  row[importanceField] = String(clampZeroToTen(importance));
  row[effectsField] = effects.map((effect) => effect.summary).join(" | ");
  historySection.rows.push(row);
};

function App() {
  const [fileName, setFileName] = useState<string>("");
  const [sheet, setSheet] = useState<CharacterSheet | null>(null);
  const [situation, setSituation] = useState<string>(DEFAULT_SITUATION);
  const [selectedModel, setSelectedModel] = useState<string>(WEBLLM_MODELS[0].id);
  const [statusText, setStatusText] = useState<string>("Load a character sheet CSV.");
  const [modelReady, setModelReady] = useState<boolean>(false);
  const [isLoadingModel, setIsLoadingModel] = useState<boolean>(false);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [errorText, setErrorText] = useState<string>("");
  const [isGpuCheckComplete, setIsGpuCheckComplete] = useState<boolean>(false);
  const [webGpuCompatible, setWebGpuCompatible] = useState<boolean>(false);
  const [gpuCheckReason, setGpuCheckReason] = useState<string>("");
  const [adapterWorkgroupStorageLimit, setAdapterWorkgroupStorageLimit] =
    useState<number | null>(null);
  const [iosLiteStatus, setIosLiteStatus] = useState<
    "idle" | "loading" | "ready" | "failed"
  >("idle");
  const [iosLiteMessage, setIosLiteMessage] = useState<string>("");
  const [generatedOutput, setGeneratedOutput] =
    useState<GeneratedNarrativeOutput | null>(null);
  const [canApplyGeneratedChanges, setCanApplyGeneratedChanges] =
    useState<boolean>(false);
  const [debugPromptSnapshot, setDebugPromptSnapshot] =
    useState<DebugPromptSnapshot | null>(null);

  const engineRef = useRef<MLCEngineInterface | null>(null);
  const iosLiteGeneratorRef = useRef<IosLiteGenerator | null>(null);

  const gpuApi =
    typeof navigator !== "undefined"
      ? (navigator as NavigatorWithGpu).gpu
      : undefined;
  const webGpuAvailable = Boolean(gpuApi);
  const debugPromptInspectorEnabled =
    typeof window !== "undefined" &&
    (() => {
      const params = new URLSearchParams(window.location.search);
      return (
        params.get("debug_prompt") === "1" ||
        params.get("debug_prompt_inspector") === "1"
      );
    })();
  const forceIosLiteMode =
    typeof window !== "undefined" &&
    new URLSearchParams(window.location.search).get("force_ios_lite") === "1";
  const isIOS = detectIOS() || forceIosLiteMode;
  const iosLowGpuLimit =
    isIOS &&
    (forceIosLiteMode ||
      (adapterWorkgroupStorageLimit !== null &&
        adapterWorkgroupStorageLimit < REQUIRED_WORKGROUP_STORAGE_SIZE));
  const iosLiteModeActive = iosLowGpuLimit && !webGpuCompatible;

  const identifyingPreview = useMemo(
    () => (sheet ? buildIdentifyingPreview(sheet) : []),
    [sheet],
  );

  const ensureIosLiteGenerator = async (): Promise<IosLiteGenerator> => {
    if (iosLiteGeneratorRef.current) {
      return iosLiteGeneratorRef.current;
    }

    setIosLiteStatus("loading");
    setIosLiteMessage(
      "Loading iOS lite local model (first run may take a while)...",
    );

    try {
      const transformers = (await import("@huggingface/transformers")) as {
        pipeline: (
          task: string,
          model: string,
        ) => Promise<IosLiteGenerator>;
      };

      const generator = await transformers.pipeline(
        "text2text-generation",
        "Xenova/flan-t5-small",
      );
      iosLiteGeneratorRef.current = generator;
      setIosLiteStatus("ready");
      setIosLiteMessage("iOS lite local model is ready.");
      return generator;
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Could not initialize iOS lite local model.";
      setIosLiteStatus("failed");
      setIosLiteMessage(message);
      throw error;
    }
  };

  const loadSheetFromText = (name: string, csvText: string) => {
    const parsedSheet = parseCharacterSheetCsv(csvText);
    const hasIdentifyingSection = Boolean(findIdentifyingSection(parsedSheet));
    setSheet(parsedSheet);
    setFileName(name);
    setGeneratedOutput(null);
    setCanApplyGeneratedChanges(false);
    setDebugPromptSnapshot(null);
    setStatusText(
      `Loaded ${name}: ${parsedSheet.sections.length} sections, ${parsedSheet.sections.reduce((sum, section) => sum + section.rows.length, 0)} rows.${hasIdentifyingSection ? "" : " No !!Identifying Traits section found."}`,
    );
    setErrorText("");
  };

  const loadDefaultTemplate = useCallback(async () => {
    try {
      const response = await fetch("./npc_char_sheet.csv");
      if (!response.ok) {
        throw new Error("Default template file not found.");
      }
      const text = await response.text();
      loadSheetFromText("npc_char_sheet.csv", text);
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Could not load default template.";
      setErrorText(message);
    }
  }, []);

  const loadTestTemplate = useCallback(async () => {
    try {
      const response = await fetch("./npc_char_sheet_test_random.csv");
      if (!response.ok) {
        throw new Error("Test sheet file not found.");
      }
      const text = await response.text();
      loadSheetFromText("npc_char_sheet_test_random.csv", text);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Could not load test sheet.";
      setErrorText(message);
    }
  }, []);

  useEffect(() => {
    void loadDefaultTemplate();
  }, [loadDefaultTemplate]);

  useEffect(() => {
    let cancelled = false;

    const checkGpuCompatibility = async () => {
      if (!webGpuAvailable) {
        if (cancelled) {
          return;
        }

        setWebGpuCompatible(false);
        setIsGpuCheckComplete(true);
        setGpuCheckReason("WebGPU is not available in this browser.");
        return;
      }

      try {
        const adapter = await gpuApi?.requestAdapter();
        if (!adapter) {
          if (cancelled) {
            return;
          }

          setWebGpuCompatible(false);
          setIsGpuCheckComplete(true);
          setGpuCheckReason("No compatible GPU adapter was found.");
          return;
        }

        const limit = adapter.limits.maxComputeWorkgroupStorageSize;
        if (cancelled) {
          return;
        }

        setAdapterWorkgroupStorageLimit(limit);
        if (limit < REQUIRED_WORKGROUP_STORAGE_SIZE) {
          setWebGpuCompatible(false);
          setGpuCheckReason(
            `Device limit ${limit} is below required ${REQUIRED_WORKGROUP_STORAGE_SIZE}.`,
          );
        } else {
          setWebGpuCompatible(true);
          setGpuCheckReason("WebGPU adapter looks compatible.");
        }
      } catch (error) {
        if (cancelled) {
          return;
        }
        const message =
          error instanceof Error ? error.message : "GPU compatibility check failed.";
        setWebGpuCompatible(false);
        setGpuCheckReason(message);
      } finally {
        if (!cancelled) {
          setIsGpuCheckComplete(true);
        }
      }
    };

    void checkGpuCompatibility();
    return () => {
      cancelled = true;
    };
  }, [gpuApi, webGpuAvailable]);

  useEffect(() => {
    if (iosLiteModeActive && iosLiteStatus === "idle") {
      setIosLiteMessage(
        "iOS low-GPU mode detected. Using a lightweight local model instead of WebLLM.",
      );
    }
  }, [iosLiteModeActive, iosLiteStatus]);

  const handleCsvUpload = async (file: File | undefined) => {
    if (!file) {
      return;
    }

    try {
      const text = await file.text();
      loadSheetFromText(file.name, text);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Could not parse CSV.";
      setErrorText(message);
      setStatusText("CSV parsing failed.");
    }
  };

  const handleLoadModel = async () => {
    if (!webGpuAvailable) {
      setErrorText("WebGPU is unavailable. Use iOS-lite or fallback path.");
      return;
    }

    if (!isGpuCheckComplete) {
      setErrorText("Still checking WebGPU compatibility. Try again in a moment.");
      return;
    }

    if (!webGpuCompatible) {
      setErrorText(
        iosLiteModeActive
          ? `Standard WebLLM is unavailable. ${gpuCheckReason} iOS lite mode will be used during generation.`
          : `Standard WebLLM is unavailable: ${gpuCheckReason}`,
      );
      return;
    }

    setErrorText("");
    setStatusText(`Loading ${selectedModel}...`);
    setIsLoadingModel(true);
    setModelReady(false);

    try {
      const { CreateMLCEngine } = await import("@mlc-ai/web-llm");
      const engine = await CreateMLCEngine(selectedModel, {
        initProgressCallback: (report) => {
          setStatusText(`${report.text} (${Math.round(report.progress * 100)}%)`);
        },
      });
      engineRef.current = engine;
      setModelReady(true);
      setStatusText(`Model ${selectedModel} is ready.`);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to load model.";
      setErrorText(message);
      setStatusText("Model load failed. Using fallback paths.");
    } finally {
      setIsLoadingModel(false);
    }
  };

  const runWebLlmGeneration = async (
    promptBundle: WebLlmPromptBundle,
    cleanSituation: string,
  ): Promise<GeneratedNarrativeOutput | null> => {
    if (!engineRef.current || !modelReady) {
      return null;
    }

    const response = await engineRef.current.chat.completions.create({
      temperature: 0,
      max_tokens: 420,
      messages: [
        { role: "system", content: promptBundle.systemPrompt },
        { role: "user", content: promptBundle.userPrompt },
      ],
    });

    const text = coerceContent(response.choices[0]?.message?.content).trim();
    const structured = parseStructuredOutput(text);
    if (structured) {
      return {
        narrative: structured.narrative,
        importance: structured.importance,
        effects: normalizeEffects(structured.effects),
      };
    }

    const narrativeFallback = extractNarrativeFromRawModelText(text);
    if (!narrativeFallback) {
      return null;
    }
    return {
      narrative: enforceNarrativeStyle(narrativeFallback, cleanSituation),
      importance: 5,
      effects: [],
    };
  };

  const runWebLlmPersonaPriming = async (
    personaSeed: string,
  ): Promise<PersonaPrimeResult> => {
    const primeBundle = buildPersonaPrimePromptBundle(personaSeed);
    if (!engineRef.current || !modelReady) {
      return {
        personaLock: buildFallbackPersonaLock(personaSeed),
        primeSystemPrompt: primeBundle.systemPrompt,
        primeUserPrompt: primeBundle.userPrompt,
        rawModelOutput: "WebLLM unavailable; using fallback persona lock.",
        source: "fallback",
      };
    }

    try {
      const response = await engineRef.current.chat.completions.create({
        temperature: 0,
        max_tokens: 280,
        messages: [
          { role: "system", content: primeBundle.systemPrompt },
          { role: "user", content: primeBundle.userPrompt },
        ],
      });
      const raw = coerceContent(response.choices[0]?.message?.content).trim();
      const parsedPersonaLock = parsePersonaPrimeOutput(raw) ?? "";
      const personaLock = isPersonaLockUsable(parsedPersonaLock)
        ? parsedPersonaLock
        : buildFallbackPersonaLock(personaSeed);

      return {
        personaLock,
        primeSystemPrompt: primeBundle.systemPrompt,
        primeUserPrompt: primeBundle.userPrompt,
        rawModelOutput: raw,
        source: "webllm",
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : "WebLLM persona priming failed.";
      return {
        personaLock: buildFallbackPersonaLock(personaSeed),
        primeSystemPrompt: primeBundle.systemPrompt,
        primeUserPrompt: primeBundle.userPrompt,
        rawModelOutput: message,
        source: "fallback",
      };
    }
  };

  const runIosLiteGeneration = async (
    litePrompt: string,
    cleanSituation: string,
  ): Promise<GeneratedNarrativeOutput | null> => {
    if (!iosLiteModeActive) {
      return null;
    }

    const generator = await ensureIosLiteGenerator();

    const result = await generator(litePrompt, {
      max_new_tokens: 200,
      do_sample: false,
      temperature: 0,
    });
    const raw = result[0]?.generated_text?.trim() ?? "";
    const structured = parseStructuredOutput(raw);
    if (structured) {
      return {
        narrative: structured.narrative,
        importance: structured.importance,
        effects: normalizeEffects(structured.effects),
      };
    }

    const narrativeFallback = extractNarrativeFromRawModelText(raw);
    if (!narrativeFallback) {
      return null;
    }
    return {
      narrative: enforceNarrativeStyle(narrativeFallback, cleanSituation),
      importance: 5,
      effects: [],
    };
  };

  const runIosLitePersonaPriming = async (
    personaSeed: string,
  ): Promise<PersonaPrimeResult> => {
    const primeBundle = buildPersonaPrimePromptBundle(personaSeed);
    if (!iosLiteModeActive) {
      return {
        personaLock: buildFallbackPersonaLock(personaSeed),
        primeSystemPrompt: primeBundle.systemPrompt,
        primeUserPrompt: primeBundle.userPrompt,
        rawModelOutput: "iOS-lite unavailable; using fallback persona lock.",
        source: "fallback",
      };
    }

    try {
      const generator = await ensureIosLiteGenerator();
      const primePrompt = [primeBundle.systemPrompt, "", primeBundle.userPrompt].join("\n");
      const result = await generator(primePrompt, {
        max_new_tokens: 220,
        do_sample: false,
        temperature: 0,
      });
      const raw = result[0]?.generated_text?.trim() ?? "";
      const parsedPersonaLock = parsePersonaPrimeOutput(raw) ?? "";
      const personaLock = isPersonaLockUsable(parsedPersonaLock)
        ? parsedPersonaLock
        : buildFallbackPersonaLock(personaSeed);
      return {
        personaLock,
        primeSystemPrompt: primeBundle.systemPrompt,
        primeUserPrompt: primeBundle.userPrompt,
        rawModelOutput: raw,
        source: "ios-lite",
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : "iOS-lite persona priming failed.";
      return {
        personaLock: buildFallbackPersonaLock(personaSeed),
        primeSystemPrompt: primeBundle.systemPrompt,
        primeUserPrompt: primeBundle.userPrompt,
        rawModelOutput: message,
        source: "fallback",
      };
    }
  };

  const handleGenerateAction = async () => {
    if (!sheet) {
      setErrorText("Load a character sheet first.");
      return;
    }
    if (!situation.trim()) {
      setErrorText("Enter a situation prompt first.");
      return;
    }

    setErrorText("");
    setIsGenerating(true);
    setStatusText("Generating narrative output...");

    const people = parseTaggedValues(situation, /#([^#\n]+?)#/g);
    const items = parseTaggedValues(situation, /\*([^*\n]+?)\*/g);
    const traitProfile = buildTraitDecisionProfile(sheet, situation, people, items);
    const personaSeed = buildPersonaSeedFromSheet(sheet, traitProfile);
    const actionPlan = buildDeterministicActionPlan(
      sheet,
      situation,
      people,
      items,
      traitProfile,
    );
    const tagEffects = buildTagEffects(sheet, people, items);
    const heuristicTraitEffects = buildHeuristicEffects(sheet, situation);
    const heuristic = buildHeuristicNarrative(actionPlan);
    const activePath: DebugPromptSnapshot["activePath"] =
      modelReady && engineRef.current
        ? "webllm"
        : iosLiteModeActive
          ? "ios-lite"
          : "heuristic-fallback";

    try {
      let generated: GeneratedNarrativeOutput | null = null;
      let personaPrimeResult: PersonaPrimeResult = {
        personaLock: buildFallbackPersonaLock(personaSeed),
        primeSystemPrompt: "",
        primeUserPrompt: "",
        rawModelOutput: "Default fallback persona lock.",
        source: "fallback",
      };

      if (modelReady && engineRef.current) {
        personaPrimeResult = await runWebLlmPersonaPriming(personaSeed);
      } else if (iosLiteModeActive) {
        personaPrimeResult = await runIosLitePersonaPriming(personaSeed);
      }

      const webLlmPromptBundle = buildWebLlmPromptBundle(
        situation.trim(),
        personaPrimeResult.personaLock,
        actionPlan,
      );
      const iosLitePromptText = buildIosLitePromptText(
        situation.trim(),
        personaPrimeResult.personaLock,
        actionPlan,
      );
      if (debugPromptInspectorEnabled) {
        setDebugPromptSnapshot({
          activePath,
          cleanSituation: traitProfile.cleanSituation,
          tags: traitProfile.tags,
          profileText: traitProfile.profileText,
          actionPlan: formatActionPlan(actionPlan),
          personaSeed,
          personaPrimeSystemPrompt: personaPrimeResult.primeSystemPrompt,
          personaPrimeUserPrompt: personaPrimeResult.primeUserPrompt,
          personaLock: personaPrimeResult.personaLock,
          personaPrimeOutput: personaPrimeResult.rawModelOutput,
          personaPrimeSource: personaPrimeResult.source,
          webLlmSystemPrompt: webLlmPromptBundle.systemPrompt,
          webLlmUserPrompt: webLlmPromptBundle.userPrompt,
          iosLitePrompt: iosLitePromptText,
        });
      }

      if (modelReady && engineRef.current) {
        generated = await runWebLlmGeneration(
          webLlmPromptBundle,
          traitProfile.cleanSituation,
        );
        if (
          generated &&
          isNarrativeWeak(
            generated.narrative,
            traitProfile.cleanSituation,
            actionPlan,
            people,
            items,
          )
        ) {
          const retryBundle: WebLlmPromptBundle = {
            systemPrompt: [
              webLlmPromptBundle.systemPrompt,
              "Retry with stricter narrative quality:",
              "- Use concrete physical actions and immediate consequences.",
              "- Do not echo wording from the situation prompt.",
              "- Do not use abstract filler phrasing.",
              ...buildActionPlanPromptAnchors(actionPlan),
            ].join("\n"),
            userPrompt: webLlmPromptBundle.userPrompt,
          };
          const retried = await runWebLlmGeneration(
            retryBundle,
            traitProfile.cleanSituation,
          );
          if (retried) {
            generated = retried;
          }
        }
      } else if (iosLiteModeActive) {
        setStatusText("Generating with iOS lite local model...");
        generated = await runIosLiteGeneration(
          iosLitePromptText,
          traitProfile.cleanSituation,
        );
        if (
          generated &&
          isNarrativeWeak(
            generated.narrative,
            traitProfile.cleanSituation,
            actionPlan,
            people,
            items,
          )
        ) {
          const retryPrompt = [
            iosLitePromptText,
            "",
            "Retry with stricter narrative quality:",
            "- Use concrete physical actions and immediate consequences.",
            "- Do not echo wording from the situation prompt.",
            "- Do not use abstract filler phrasing.",
            ...buildActionPlanPromptAnchors(actionPlan),
          ].join("\n");
          const retried = await runIosLiteGeneration(
            retryPrompt,
            traitProfile.cleanSituation,
          );
          if (retried) {
            generated = retried;
          }
        }
      }

      if (
        generated &&
        isNarrativeWeak(
          generated.narrative,
          traitProfile.cleanSituation,
          actionPlan,
          people,
          items,
        )
      ) {
        generated = null;
        setStatusText(
          "Model output was too generic or echoed the prompt; using deterministic fallback narrative.",
        );
      }

      const finalGenerated = generated ?? heuristic;
      const stylizedNarrative = enforceNarrativeStyle(
        finalGenerated.narrative,
        traitProfile.cleanSituation,
      );
      const combinedEffects = [
        ...tagEffects,
        ...finalGenerated.effects,
        ...heuristicTraitEffects,
      ];
      const dedupedEffects = combinedEffects.filter(
        (effect, index, all) =>
          all.findIndex(
            (candidate) =>
              normalizeName(candidate.summary) === normalizeName(effect.summary),
          ) === index,
      );

      setGeneratedOutput({
        narrative: stylizedNarrative,
        importance: finalGenerated.importance,
        effects: dedupedEffects,
      });
      setCanApplyGeneratedChanges(true);
      if (generated && modelReady) {
        setStatusText("Generated narrative and effects with WebLLM.");
      } else if (generated && iosLiteModeActive) {
        setStatusText("Generated narrative and effects with iOS lite model.");
      } else {
        setStatusText("Generated heuristic narrative and effects fallback.");
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Generation failed.";
      setErrorText(message);

      const fallbackOutput: GeneratedNarrativeOutput = {
        ...heuristic,
        effects: [...tagEffects, ...heuristic.effects, ...heuristicTraitEffects],
      };
      setGeneratedOutput(fallbackOutput);
      setCanApplyGeneratedChanges(true);
      setStatusText("Generation failed; using deterministic fallback output.");
    } finally {
      setIsGenerating(false);
    }
  };

  const applyGeneratedChanges = () => {
    if (!sheet || !generatedOutput) {
      return;
    }

    const nextSheet = cloneSheet(sheet);
    generatedOutput.effects.forEach((effect) => applyEffectToSheet(nextSheet, effect));
    appendHistoryEvent(
      nextSheet,
      generatedOutput.narrative,
      generatedOutput.importance,
      generatedOutput.effects,
    );

    setSheet(nextSheet);
    setCanApplyGeneratedChanges(false);
    setStatusText(
      `Applied ${generatedOutput.effects.length} effects and appended history event.`,
    );
  };

  const downloadUpdatedSheet = () => {
    if (!sheet) {
      return;
    }
    const csvText = serializeCharacterSheetCsv(sheet);
    const blob = new Blob([csvText], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = fileName
      ? fileName.replace(/\.csv$/i, "_updated.csv")
      : "npc_char_sheet_updated.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="app">
      <section className="card">
        <h1>NPC Action Generator</h1>
        <p className="subtitle">
          Load sheet only: section/field CSV format using !! and ! markers.
        </p>

        <label className="label" htmlFor="csv-upload">
          Character sheet CSV
        </label>
        <input
          id="csv-upload"
          type="file"
          accept=".csv,text/csv"
          onChange={(event) => handleCsvUpload(event.target.files?.[0])}
        />
        <button className="button secondary" type="button" onClick={loadDefaultTemplate}>
          Load default template file
        </button>
        <button className="button secondary" type="button" onClick={loadTestTemplate}>
          Load random test sheet
        </button>
        <button className="button secondary" type="button" onClick={downloadUpdatedSheet} disabled={!sheet}>
          Download updated sheet
        </button>

        <p className="meta">
          {fileName ? `Loaded file: ${fileName}` : "No file loaded."}
        </p>
        <p className="meta">
          Sections: <strong>{sheet?.sections.length ?? 0}</strong>
        </p>

        <label className="label">
          Identifying traits preview (first 5 from !!Identifying Traits)
        </label>
        {identifyingPreview.length > 0 ? (
          <ul className="trait-list">
            {identifyingPreview.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        ) : (
          <p className="meta">No identifying traits found yet.</p>
        )}
      </section>

      <section className="card">
        <label className="label" htmlFor="model-select">
          Model
        </label>
        <select
          id="model-select"
          value={selectedModel}
          onChange={(event) => setSelectedModel(event.target.value)}
        >
          {WEBLLM_MODELS.map((model) => (
            <option key={model.id} value={model.id}>
              {model.label}
            </option>
          ))}
        </select>
        <button
          className="button"
          type="button"
          onClick={handleLoadModel}
          disabled={
            isLoadingModel ||
            !webGpuAvailable ||
            !isGpuCheckComplete ||
            !webGpuCompatible
          }
        >
          {!isGpuCheckComplete
            ? "Checking GPU..."
            : !webGpuCompatible
              ? "Load Model (unsupported)"
              : isLoadingModel
                ? "Loading model..."
                : "Load Model"}
        </button>
        <p className="meta">
          WebGPU support: <strong>{webGpuAvailable ? "available" : "not detected"}</strong>
        </p>
        <p className="meta">
          GPU limit (workgroup storage):{" "}
          <strong>
            {adapterWorkgroupStorageLimit === null ? "unknown" : adapterWorkgroupStorageLimit}
          </strong>
        </p>
        <p className="meta">
          Runtime requirement: <strong>{REQUIRED_WORKGROUP_STORAGE_SIZE}</strong>
        </p>
        <p className="meta">
          WebLLM compatibility: <strong>{webGpuCompatible ? "compatible" : "not compatible"}</strong>
        </p>
        {gpuCheckReason && <p className="meta">{gpuCheckReason}</p>}
        {iosLiteModeActive && (
          <>
            <p className="meta">
              iOS lite local mode:{" "}
              <strong>
                {iosLiteStatus === "ready"
                  ? "ready"
                  : iosLiteStatus === "loading"
                    ? "loading"
                    : iosLiteStatus === "failed"
                      ? "failed"
                      : "available"}
              </strong>
            </p>
            {iosLiteMessage && <p className="meta">{iosLiteMessage}</p>}
          </>
        )}
        <p className="meta">
          LLM status: <strong>{modelReady ? "ready" : "not loaded"}</strong>
        </p>

        <label className="label" htmlFor="situation-input">
          Situation prompt
        </label>
        <textarea
          id="situation-input"
          rows={6}
          value={situation}
          placeholder="Use #Name# for connections and *item* for inventory references."
          onChange={(event) => setSituation(event.target.value)}
        />
        <button
          className="button primary"
          type="button"
          onClick={handleGenerateAction}
          disabled={isGenerating}
        >
          {isGenerating ? "Generating..." : "Generate Narrative + Effects"}
        </button>
        <button
          className="button"
          type="button"
          onClick={applyGeneratedChanges}
          disabled={!generatedOutput || !canApplyGeneratedChanges}
        >
          Apply listed changes to sheet
        </button>

        {errorText && <p className="error">{errorText}</p>}
        <p className="status">{statusText}</p>

        <label className="label">Narrative output</label>
        <textarea
          rows={6}
          readOnly
          value={generatedOutput?.narrative ?? ""}
          className="output-box"
        />
        <p className="meta">
          Event importance: <strong>{generatedOutput?.importance ?? "-"}</strong>
        </p>

        <label className="label">Effects</label>
        {generatedOutput && generatedOutput.effects.length > 0 ? (
          <ul className="effect-list">
            {generatedOutput.effects.map((effect) => (
              <li key={effect.id}>{effect.summary}</li>
            ))}
          </ul>
        ) : (
          <p className="meta">No effects listed yet.</p>
        )}

        {debugPromptInspectorEnabled && (
          <details className="debug-panel">
            <summary>Debug prompt inspector (?debug_prompt=1)</summary>
            <p className="meta">
              Active generation path:{" "}
              <strong>{debugPromptSnapshot?.activePath ?? "not generated yet"}</strong>
            </p>
            <p className="meta">
              Situation used:{" "}
              <strong>{debugPromptSnapshot?.cleanSituation ?? "Generate once to inspect."}</strong>
            </p>
            <p className="meta">
              Situation tags:{" "}
              <strong>
                {debugPromptSnapshot
                  ? debugPromptSnapshot.tags.join(", ") || "general"
                  : "Generate once to inspect."}
              </strong>
            </p>
            <p className="meta">
              Persona prime source:{" "}
              <strong>{debugPromptSnapshot?.personaPrimeSource ?? "not generated yet"}</strong>
            </p>

            <label className="label">Persona seed (pre-situation)</label>
            <textarea
              rows={8}
              readOnly
              className="debug-output"
              value={debugPromptSnapshot?.personaSeed ?? ""}
            />

            <label className="label">Persona prime system prompt</label>
            <textarea
              rows={7}
              readOnly
              className="debug-output"
              value={debugPromptSnapshot?.personaPrimeSystemPrompt ?? ""}
            />

            <label className="label">Persona prime user prompt</label>
            <textarea
              rows={8}
              readOnly
              className="debug-output"
              value={debugPromptSnapshot?.personaPrimeUserPrompt ?? ""}
            />

            <label className="label">Persona lock generated before situation</label>
            <textarea
              rows={8}
              readOnly
              className="debug-output"
              value={debugPromptSnapshot?.personaLock ?? ""}
            />

            <label className="label">Raw persona-prime model output</label>
            <textarea
              rows={8}
              readOnly
              className="debug-output"
              value={debugPromptSnapshot?.personaPrimeOutput ?? ""}
            />

            <label className="label">Trait profile inspector</label>
            <textarea
              rows={10}
              readOnly
              className="debug-output"
              value={debugPromptSnapshot?.profileText ?? ""}
            />

            <label className="label">Deterministic action plan (silent context)</label>
            <textarea
              rows={9}
              readOnly
              className="debug-output"
              value={debugPromptSnapshot?.actionPlan ?? ""}
            />

            <label className="label">WebLLM system prompt sent to model</label>
            <textarea
              rows={8}
              readOnly
              className="debug-output"
              value={debugPromptSnapshot?.webLlmSystemPrompt ?? ""}
            />

            <label className="label">WebLLM user prompt sent to model</label>
            <textarea
              rows={8}
              readOnly
              className="debug-output"
              value={debugPromptSnapshot?.webLlmUserPrompt ?? ""}
            />

            <label className="label">iOS-lite prompt sent to model</label>
            <textarea
              rows={8}
              readOnly
              className="debug-output"
              value={debugPromptSnapshot?.iosLitePrompt ?? ""}
            />
          </details>
        )}
      </section>

      <section className="card hint">
        <h2>Sheet format rules</h2>
        <p>
          <strong>!!</strong> starts a section. Rows below belong to that section until next !!
          .
        </p>
        <p>
          <strong>!</strong> marks field header cells. Data rows below map to those fields.
        </p>
        <p>
          Prompt tags: <code>#Name#</code> for connections, <code>*item*</code> for inventory.
        </p>
        <p>Use "Load random test sheet" for quick end-to-end testing.</p>
        <p>
          Generated output includes narrative + effects list. Apply button writes those effects
          to the sheet and appends a History event with importance and effect summary.
        </p>
      </section>
    </main>
  );
}

export default App;
