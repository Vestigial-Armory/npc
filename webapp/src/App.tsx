import { useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import type { MLCEngineInterface } from "@mlc-ai/web-llm";
import "./App.css";

type CsvRow = Record<string, string>;

type WeightedTrait = {
  value: string;
  weight: number;
};

type TraitTable = Record<string, WeightedTrait[]>;

type RolledTrait = {
  trait: string;
  value: string;
};

const MODELS = [
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

const normalize = (value: string) => value.trim().toLowerCase();

const pickWeighted = (items: WeightedTrait[]): string => {
  const total = items.reduce((sum, item) => sum + item.weight, 0);
  if (total <= 0) {
    return items[0]?.value ?? "unknown";
  }

  let target = Math.random() * total;
  for (const item of items) {
    target -= item.weight;
    if (target <= 0) {
      return item.value;
    }
  }

  return items[items.length - 1]?.value ?? "unknown";
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

const fallbackAction = (situation: string, rolledTraits: RolledTrait[]): string => {
  const traitText =
    rolledTraits.length > 0
      ? rolledTraits.map((entry) => `${entry.trait}: ${entry.value}`).join(", ")
      : "no specific rolled traits";

  return `Rules-only fallback action: Given ${traitText}, the NPC addresses this situation "${situation}" by taking a cautious first step, gathering one extra detail, and then committing to a single clear action that fits their strongest motivation.`;
};

function App() {
  const [fileName, setFileName] = useState<string>("");
  const [traitTable, setTraitTable] = useState<TraitTable>({});
  const [rolledTraits, setRolledTraits] = useState<RolledTrait[]>([]);
  const [situation, setSituation] = useState<string>("");
  const [actionOutput, setActionOutput] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<string>(MODELS[0].id);
  const [statusText, setStatusText] = useState<string>(
    "Upload a CSV to start.",
  );
  const [modelReady, setModelReady] = useState<boolean>(false);
  const [isLoadingModel, setIsLoadingModel] = useState<boolean>(false);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [errorText, setErrorText] = useState<string>("");
  const engineRef = useRef<MLCEngineInterface | null>(null);

  const webGpuAvailable =
    typeof navigator !== "undefined" && "gpu" in navigator;

  const traitCount = useMemo(
    () => Object.keys(traitTable).length,
    [traitTable],
  );

  const parseCsvToTraitTable = (rows: CsvRow[]) => {
    const nextTable: TraitTable = {};

    for (const row of rows) {
      const keys = Object.keys(row);
      if (keys.length === 0) {
        continue;
      }

      const trait =
        row.trait ??
        row.category ??
        row.attribute ??
        row[keys[0]] ??
        "";

      const value =
        row.value ??
        row.option ??
        row.result ??
        (keys[1] ? row[keys[1]] : "") ??
        "";

      const weightInput =
        row.weight ?? row.probability ?? row.chance ?? row.percent ?? "1";
      const weight = Number.parseFloat(weightInput);

      const cleanTrait = trait.trim();
      const cleanValue = value.trim();
      if (!cleanTrait || !cleanValue) {
        continue;
      }

      if (!nextTable[cleanTrait]) {
        nextTable[cleanTrait] = [];
      }

      nextTable[cleanTrait].push({
        value: cleanValue,
        weight: Number.isFinite(weight) && weight > 0 ? weight : 1,
      });
    }

    return nextTable;
  };

  const handleCsvUpload = async (file: File | undefined) => {
    if (!file) {
      return;
    }

    setErrorText("");
    setActionOutput("");
    setStatusText("Parsing CSV...");

    try {
      const csvText = await file.text();
      const result = Papa.parse<CsvRow>(csvText, {
        header: true,
        skipEmptyLines: true,
        transformHeader: (header) => normalize(header),
      });

      if (result.errors.length > 0) {
        throw new Error(result.errors[0]?.message ?? "Invalid CSV format.");
      }

      const nextTable = parseCsvToTraitTable(result.data);
      const parsedTraits = Object.keys(nextTable).length;

      if (parsedTraits === 0) {
        throw new Error(
          "No trait rows found. CSV should include trait/category + value + optional weight columns.",
        );
      }

      setFileName(file.name);
      setTraitTable(nextTable);
      setRolledTraits([]);
      setStatusText(`Loaded ${parsedTraits} trait groups from ${file.name}.`);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Could not parse CSV.";
      setErrorText(message);
      setStatusText("CSV parsing failed.");
    }
  };

  const handleRollTraits = () => {
    setErrorText("");
    setActionOutput("");
    const nextRoll = Object.entries(traitTable).map(([trait, options]) => ({
      trait,
      value: pickWeighted(options),
    }));
    setRolledTraits(nextRoll);
    setStatusText(`Rolled ${nextRoll.length} traits.`);
  };

  const handleLoadModel = async () => {
    if (!webGpuAvailable) {
      setErrorText(
        "WebGPU is unavailable on this device/browser. Use rules-only fallback mode.",
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
          const progressPercent = Math.round(report.progress * 100);
          setStatusText(`${report.text} (${progressPercent}%)`);
        },
      });

      engineRef.current = engine;
      setModelReady(true);
      setStatusText(`Model ${selectedModel} is ready.`);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to load model.";
      setErrorText(message);
      setStatusText("Model load failed. You can still use fallback mode.");
      setModelReady(false);
    } finally {
      setIsLoadingModel(false);
    }
  };

  const handleGenerateAction = async () => {
    if (!situation.trim()) {
      setErrorText("Enter a situation prompt first.");
      return;
    }

    setErrorText("");
    setIsGenerating(true);
    setStatusText("Generating NPC action...");

    if (!engineRef.current || !modelReady) {
      const fallback = fallbackAction(situation, rolledTraits);
      setActionOutput(fallback);
      setStatusText("Generated with rules-only fallback.");
      setIsGenerating(false);
      return;
    }

    const traitSummary =
      rolledTraits.length > 0
        ? rolledTraits.map((entry) => `- ${entry.trait}: ${entry.value}`).join("\n")
        : "- No trait rolls provided";

    const systemPrompt = [
      "You are an NPC action generator for a tabletop RPG.",
      "Return one concise action response (2-4 sentences).",
      "Use the NPC traits as weighted personality context.",
      "Avoid meta commentary and avoid bullet lists.",
      "NPC traits:",
      traitSummary,
    ].join("\n");

    try {
      const response = await engineRef.current.chat.completions.create({
        temperature: 0.8,
        max_tokens: 180,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: situation.trim() },
        ],
      });

      const generated = coerceContent(response.choices[0]?.message?.content).trim();
      if (!generated) {
        throw new Error("Model returned an empty action.");
      }

      setActionOutput(generated);
      setStatusText("Action generated with WebLLM.");
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Generation failed.";
      setErrorText(message);
      setActionOutput(fallbackAction(situation, rolledTraits));
      setStatusText("Generation failed; showing rules-only fallback action.");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <main className="app">
      <section className="card">
        <h1>NPC Action Generator</h1>
        <p className="subtitle">
          Mobile-first static app: CSV traits + in-browser LLM inference.
        </p>

        <label className="label" htmlFor="csv-upload">
          Trait CSV
        </label>
        <input
          id="csv-upload"
          type="file"
          accept=".csv,text/csv"
          onChange={(event) => handleCsvUpload(event.target.files?.[0])}
        />
        <p className="meta">
          {fileName ? `Loaded file: ${fileName}` : "No CSV selected yet."}
        </p>
        <p className="meta">
          Parsed trait groups: <strong>{traitCount}</strong>
        </p>
        <button
          className="button"
          type="button"
          onClick={handleRollTraits}
          disabled={traitCount === 0}
        >
          Roll NPC Traits
        </button>

        {rolledTraits.length > 0 && (
          <ul className="trait-list">
            {rolledTraits.map((entry) => (
              <li key={entry.trait}>
                <strong>{entry.trait}:</strong> {entry.value}
              </li>
            ))}
          </ul>
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
          {MODELS.map((model) => (
            <option key={model.id} value={model.id}>
              {model.label}
            </option>
          ))}
        </select>
        <button
          className="button"
          type="button"
          onClick={handleLoadModel}
          disabled={isLoadingModel || !webGpuAvailable}
        >
          {isLoadingModel ? "Loading model..." : "Load Model"}
        </button>
        <p className="meta">
          WebGPU support:{" "}
          <strong>{webGpuAvailable ? "available" : "not detected"}</strong>
        </p>
        <p className="meta">
          LLM status: <strong>{modelReady ? "ready" : "not loaded"}</strong>
        </p>

        <label className="label" htmlFor="situation-input">
          Situation prompt
        </label>
        <textarea
          id="situation-input"
          rows={5}
          value={situation}
          placeholder="Example: The party refuses to pay the ferryman after crossing the river."
          onChange={(event) => setSituation(event.target.value)}
        />
        <button
          className="button primary"
          type="button"
          onClick={handleGenerateAction}
          disabled={isGenerating}
        >
          {isGenerating ? "Generating..." : "Generate NPC Action"}
        </button>

        {errorText && <p className="error">{errorText}</p>}
        <p className="status">{statusText}</p>

        <label className="label" htmlFor="action-output">
          Action output
        </label>
        <textarea id="action-output" rows={7} readOnly value={actionOutput} />
      </section>

      <section className="card hint">
        <h2>CSV format hint</h2>
        <p>Use columns named: trait, value, weight (weight optional).</p>
        <p>
          Quick start:{" "}
          <a href="./sample_traits.csv" download>
            download sample CSV
          </a>
        </p>
        <p>Example rows:</p>
        <pre>{`trait,value,weight
personality,curious,3
personality,suspicious,1
motivation,protect family,4`}</pre>
      </section>
    </main>
  );
}

export default App;
