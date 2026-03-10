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
  for (const match of text.matchAll(regex)) {
    const value = (match[1] ?? "").trim();
    if (!value) {
      continue;
    }
    const key = normalizeName(value);
    if (!seen.has(key)) {
      seen.add(key);
    }
  }
  return Array.from(seen.values());
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
  const identifyingSection =
    findSection(sheet, ["identifying", "identity", "profile"]) ??
    sheet.sections.find((section) => section.rows.length > 0);

  if (!identifyingSection) {
    return [];
  }

  const traitField =
    findFieldByKeywords(identifyingSection.headers, [
      "trait",
      "name",
      "identifier",
      "label",
    ]) ?? identifyingSection.headers[0];

  const valueField =
    findFieldByKeywords(identifyingSection.headers, ["value", "rating"]) ??
    identifyingSection.headers[1];

  const emphasisField = findFieldByKeywords(identifyingSection.headers, [
    "emphasis",
    "weight",
    "importance",
  ]);

  return identifyingSection.rows.slice(0, 5).map((row) => {
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

const buildHeuristicEffects = (
  sheet: CharacterSheet,
  prompt: string,
): CharacterOutputEffect[] => {
  const dangerKeywords = ["attack", "threat", "fight", "ambush", "betray"];
  const calmKeywords = ["gift", "help", "peace", "apology", "alliance"];
  const lowerPrompt = normalizeName(prompt);
  const identifyingSection =
    findSection(sheet, ["identifying", "identity", "profile"]) ??
    sheet.sections.find((section) => section.rows.length > 0);

  if (!identifyingSection || identifyingSection.rows.length === 0) {
    return [];
  }

  const traitField =
    findFieldByKeywords(identifyingSection.headers, ["trait", "name"]) ??
    identifyingSection.headers[0];
  const emphasisField = findFieldByKeywords(identifyingSection.headers, [
    "emphasis",
    "importance",
  ]);

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
  previewTraits: string[],
): GeneratedNarrativeOutput => {
  const traitSummary =
    previewTraits.length > 0
      ? previewTraits.join("; ")
      : "their core traits and motives";

  return {
    narrative: `The NPC reacts to the situation by balancing caution with initiative. Guided by ${traitSummary}, they make a clear move that addresses immediate risk, then adjust their stance for the next moment.`,
    importance: 6,
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

const buildSheetSummaryForPrompt = (sheet: CharacterSheet): string =>
  sheet.sections
    .map((section) => {
      const previewRows = section.rows.slice(0, 4).map((row) => {
        const pairs = section.headers
          .slice(0, 5)
          .map((header) => `${header}: ${row[header] ?? ""}`)
          .join(", ");
        return `  - ${pairs}`;
      });
      return [`Section: ${section.name}`, ...previewRows].join("\n");
    })
    .join("\n");

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

  const engineRef = useRef<MLCEngineInterface | null>(null);
  const iosLiteGeneratorRef = useRef<IosLiteGenerator | null>(null);

  const gpuApi =
    typeof navigator !== "undefined"
      ? (navigator as NavigatorWithGpu).gpu
      : undefined;
  const webGpuAvailable = Boolean(gpuApi);
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
    setSheet(parsedSheet);
    setFileName(name);
    setGeneratedOutput(null);
    setCanApplyGeneratedChanges(false);
    setStatusText(
      `Loaded ${name}: ${parsedSheet.sections.length} sections, ${parsedSheet.sections.reduce((sum, section) => sum + section.rows.length, 0)} rows.`,
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
    promptText: string,
    sheetSummary: string,
  ): Promise<GeneratedNarrativeOutput | null> => {
    if (!engineRef.current || !modelReady) {
      return null;
    }

    const systemPrompt = [
      "You are an NPC action engine.",
      "Return strict JSON only.",
      "JSON schema:",
      '{"narrative":"string","importance":0-10,"effects":[{"kind":"add_row|adjust_field|set_field|remove_row|note","section":"string","matchField":"string","matchValue":"string","field":"string","delta":-3..3,"value":"string","row":{"field":"value"},"summary":"string"}]}',
      "Effects should be short, concrete, and relevant.",
      "Narrative must be immersive and concise.",
      "Character sheet summary:",
      sheetSummary,
    ].join("\n");

    const response = await engineRef.current.chat.completions.create({
      temperature: 0.7,
      max_tokens: 320,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: promptText },
      ],
    });

    const text = coerceContent(response.choices[0]?.message?.content).trim();
    const structured = parseStructuredOutput(text);
    if (!structured) {
      return null;
    }
    return {
      narrative: structured.narrative,
      importance: structured.importance,
      effects: normalizeEffects(structured.effects),
    };
  };

  const runIosLiteGeneration = async (
    promptText: string,
    sheetSummary: string,
  ): Promise<GeneratedNarrativeOutput | null> => {
    if (!iosLiteModeActive) {
      return null;
    }

    const generator = await ensureIosLiteGenerator();
    const litePrompt = [
      "Return JSON only with keys narrative, importance, effects.",
      "Narrative is 2-4 sentences.",
      "Importance is 0-10.",
      "Effects is an array with updates.",
      "Character summary:",
      sheetSummary,
      "Situation:",
      promptText,
    ].join("\n");

    const result = await generator(litePrompt, {
      max_new_tokens: 200,
      do_sample: true,
      temperature: 0.7,
    });
    const raw = result[0]?.generated_text?.trim() ?? "";
    const structured = parseStructuredOutput(raw);
    if (!structured) {
      return null;
    }
    return {
      narrative: structured.narrative,
      importance: structured.importance,
      effects: normalizeEffects(structured.effects),
    };
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
    const tagEffects = buildTagEffects(sheet, people, items);
    const heuristicTraitEffects = buildHeuristicEffects(sheet, situation);
    const preview = buildIdentifyingPreview(sheet);
    const heuristic = buildHeuristicNarrative(preview);
    const sheetSummary = buildSheetSummaryForPrompt(sheet);

    try {
      let generated: GeneratedNarrativeOutput | null = null;

      if (modelReady && engineRef.current) {
        generated = await runWebLlmGeneration(situation.trim(), sheetSummary);
      } else if (iosLiteModeActive) {
        setStatusText("Generating with iOS lite local model...");
        generated = await runIosLiteGeneration(situation.trim(), sheetSummary);
      }

      const finalGenerated = generated ?? heuristic;
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
        narrative: finalGenerated.narrative,
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
        <button className="button secondary" type="button" onClick={downloadUpdatedSheet} disabled={!sheet}>
          Download updated sheet
        </button>

        <p className="meta">
          {fileName ? `Loaded file: ${fileName}` : "No file loaded."}
        </p>
        <p className="meta">
          Sections: <strong>{sheet?.sections.length ?? 0}</strong>
        </p>

        <label className="label">Identifying traits preview (first 5)</label>
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
        <p>
          Generated output includes narrative + effects list. Apply button writes those effects
          to the sheet and appends a History event with importance and effect summary.
        </p>
      </section>
    </main>
  );
}

export default App;
