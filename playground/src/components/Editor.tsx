import { useEffect, useRef, useState } from "preact/hooks";
import { EditorState } from "@codemirror/state";
import { EditorView, keymap, lineNumbers, highlightActiveLine } from "@codemirror/view";
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { oneDark } from "@codemirror/theme-one-dark";

// ─── Golden Responses ──────────────────────────────────────────────────────
// Pre-computed "ideal" LLM responses for each demo, keyed by prompt substring.
// Used as default so LLM demos work instantly without a model download.

type GoldenEntry = { match: string; response: string; channel?: string };

const GOLDEN_RESPONSES: GoldenEntry[] = [
  // ── Classification ──
  { match: "capital of France", response: "geography" },
  { match: "tall is Mount Everest", response: "geography" },
  { match: "wrote Hamlet", response: "literature" },

  // ── Sentiment ──
  { match: "amazing! Best purchase", response: "positive" },
  { match: "Terrible quality, broke", response: "negative" },
  { match: "nothing special but works", response: "neutral" },

  // ── Structured Extraction ──
  { match: "List only the people", response: '["Alice", "Bob", "Charlie"]' },

  // ── Summarization ──
  {
    match: "Summarize this in exactly one sentence",
    response:
      "Domain-specific languages for LLM orchestration use functional programming and built-in resource limits to make AI pipelines safer, more testable, and easier to debug than general-purpose alternatives.",
  },

  // ── Multi-Step Analysis ──
  { match: "AI safety is crucial", response: "AI safety matters more now" },
  { match: "Functional programming reduces bugs", response: "Immutability reduces software bugs" },
  { match: "DSLs can enforce constraints", response: "DSLs enforce unique constraints" },
  {
    match: "Synthesize these points",
    response:
      "Purpose-built, immutable DSLs are the safest path to reliable AI systems because they enforce constraints that general-purpose languages cannot.",
  },

  // ── Model Channels ──
  {
    match: "quantum entanglement",
    channel: "fast",
    response:
      "Two particles linked — measuring one instantly affects the other, regardless of distance.",
  },
  {
    match: "quantum entanglement",
    channel: "smart",
    response:
      "Quantum entanglement is a phenomenon where two or more particles become correlated such that the quantum state of each particle cannot be described independently. When you measure one entangled particle, you instantly determine the corresponding property of its partner, even across vast distances. This isn't communication — it's a fundamental feature of quantum mechanics that Einstein called 'spooky action at a distance.'",
  },
  // Generic fallback for quantum entanglement without channel
  {
    match: "quantum entanglement",
    response:
      "Two particles linked — measuring one instantly affects the other, regardless of distance.",
  },
];

function goldenAskHandler(prompt: string, channel: string | null): string {
  // Try channel-specific match first, then generic
  const candidates = GOLDEN_RESPONSES.filter((g) => prompt.includes(g.match));
  if (channel) {
    const channelMatch = candidates.find((g) => g.channel === channel);
    if (channelMatch) return channelMatch.response;
  }
  const generic = candidates.find((g) => !g.channel) || candidates[0];
  if (generic) return generic.response;
  return "(simulated LLM response)";
}

type LLMMode = "golden" | "prompt" | "wllama";


// Example programs bundled with the playground
const EXAMPLES: Record<string, { name: string; code: string; needsLLM: boolean; category: string }> = {
  // ─── Pure: Basics ───────────────────────────────────────────────────────
  hello: {
    name: "Filter & Transform",
    category: "Pure: Basics",
    needsLLM: false,
    code: `-- Filter and transform a list
let fruits = ["apple", "banana", "avocado", "blueberry", "cherry", "apricot"]
let a_fruits = filter fruits where starts_with(it, "a")
let shouted = map a_fruits with upper(it)
return "A-fruits: " ++ join shouted with ", "`,
  },
  pipeline: {
    name: "Pipeline Operator",
    category: "Pure: Basics",
    needsLLM: false,
    code: `-- Extract errors from a log using |> pipeline
let log = "ERROR: disk full
INFO: backup started
ERROR: network timeout
INFO: backup complete
WARN: low memory
ERROR: auth failed"

let errors = log
    |> lines
    |> filter where starts_with(it, "ERROR")
    |> map with replace(it, "ERROR: ", "")

return "Found " ++ show(length(errors)) ++ " errors: " ++ join errors with ", "`,
  },
  records: {
    name: "Records & Fields",
    category: "Pure: Basics",
    needsLLM: false,
    code: `-- Work with structured data using records
let users = [
    { name: "Alice", age: 30, role: "Engineer" },
    { name: "Bob", age: 25, role: "Designer" },
    { name: "Charlie", age: 35, role: "Engineer" }
]

let engineers = filter users where it.role == "Engineer"
let names = map engineers with it.name
return "Engineers: " ++ join names with ", "`,
  },
  conditionals: {
    name: "Conditionals",
    category: "Pure: Basics",
    needsLLM: false,
    code: `-- Conditional expressions with if/then/else
let scores = [85, 92, 78, 95, 67, 88]

let grades = map scores with
    if it >= 90 then "A"
    else if it >= 80 then "B"
    else if it >= 70 then "C"
    else "F"

return "Grades: " ++ join grades with ", "`,
  },

  // ─── Pure: Strings ──────────────────────────────────────────────────────
  strings: {
    name: "String Functions",
    category: "Pure: Strings",
    needsLLM: false,
    code: `-- String manipulation functions
let text = "  Hello, World!  "

let results = [
    "trimmed: '" ++ trim(text) ++ "'",
    "upper: " ++ upper(text),
    "lower: " ++ lower(text),
    "length: " ++ show(length(trim(text))),
    "contains 'World': " ++ show(contains(text, "World")),
    "replaced: " ++ replace(text, "World", "Hoist")
]

return join results with "\\n"`,
  },
  regex: {
    name: "Regex Operations",
    category: "Pure: Strings",
    needsLLM: false,
    code: `-- Regular expression matching and extraction
let text = "Contact us at support@example.com or sales@company.org"

let emails = find_all(text, "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\.[a-zA-Z]{2,}")
let has_email = matches(text, "@")
let first_email = find(text, "[\\\\w.]+@[\\\\w.]+") or "none"

return "Emails found: " ++ join emails with ", " ++ "\\nFirst: " ++ first_email`,
  },
  splitjoin: {
    name: "Split & Join",
    category: "Pure: Strings",
    needsLLM: false,
    code: `-- Split strings and join them back
let csv = "apple,banana,cherry,date"
let items = split csv by ","
let numbered = map (enumerate items) with
    show((first(it) or 0) + 1) ++ ". " ++ (last(it) or "")

return join numbered with "\\n"`,
  },

  // ─── Pure: Collections ──────────────────────────────────────────────────
  mapreduce: {
    name: "Map-Reduce",
    category: "Pure: Collections",
    needsLLM: false,
    code: `-- Classic map-reduce pattern
let numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

let squared = map numbers with it * it
let even_squares = filter squared where it % 2 == 0
let total = fold even_squares from 0 with acc + it

return "Even squares: " ++ join (map even_squares with show(it)) with ", " ++ "\\nSum: " ++ show(total)`,
  },
  zipunzip: {
    name: "Zip & Enumerate",
    category: "Pure: Collections",
    needsLLM: false,
    code: `-- Combine and index lists
let names = ["Alice", "Bob", "Charlie"]
let scores = [95, 87, 92]

let combined = zip(names, scores)
let formatted = map combined with
    (first(it) or "") ++ ": " ++ show(last(it) or 0)

return join formatted with "\\n"`,
  },
  sorting: {
    name: "Sort & Unique",
    category: "Pure: Collections",
    needsLLM: false,
    code: `-- Sort and deduplicate
let words = ["banana", "apple", "cherry", "apple", "date", "banana"]
let unique_sorted = sort(unique(words))

return "Original: " ++ join words with ", " ++
       "\\nUnique & sorted: " ++ join unique_sorted with ", "`,
  },
  window: {
    name: "Sliding Window",
    category: "Pure: Collections",
    needsLLM: false,
    code: `-- Sliding window for moving averages
let temps = [72, 75, 71, 73, 78, 76, 74]
let windows = window temps size 3

let averages = map windows with
    let sum = fold it from 0 with acc + it
    sum / length(it)

return "Temperatures: " ++ join (map temps with show(it)) with ", " ++
       "\\n3-day moving avg: " ++ join (map averages with show(it)) with ", "`,
  },

  // ─── Pure: Advanced ─────────────────────────────────────────────────────
  optionals: {
    name: "Optional Values",
    category: "Pure: Advanced",
    needsLLM: false,
    code: `-- Safe handling of optional values
let users = [
    { name: "Alice", email: "alice@example.com" },
    { name: "Bob" },
    { name: "Charlie", email: "charlie@example.com" }
]

let emails = map users with
    let email = first(filter [it] where it.email?) or { email: "N/A" }
    it.name ++ ": " ++ (email.email or "no email")

return join emails with "\\n"`,
  },
  nested: {
    name: "Nested Pipelines",
    category: "Pure: Advanced",
    needsLLM: false,
    code: `-- Complex nested transformations
let data = [
    { dept: "Engineering", members: ["Alice", "Bob"] },
    { dept: "Design", members: ["Charlie"] },
    { dept: "Engineering", members: ["David", "Eve"] }
]

let eng_members = data
    |> filter where it.dept == "Engineering"
    |> map with it.members
    |> flatten
    |> sort
    |> unique

return "All Engineering: " ++ join eng_members with ", "`,
  },

  // ─── LLM: Basic ─────────────────────────────────────────────────────────
  classify: {
    name: "Classification",
    category: "LLM: Basic",
    needsLLM: true,
    code: `-- Classify items using an LLM
let questions = [
    "What is the capital of France?",
    "How tall is Mount Everest?",
    "Who wrote Hamlet?"
]

let categories = map questions with
    ask "Classify as: geography, science, history, or literature. Reply with ONE word only.\\n\\nQuestion: {it}"

let results = map (zip(questions, categories)) with
    (first(it) or "") ++ " → " ++ (last(it) or "")

return join results with "\\n"`,
  },
  sentiment: {
    name: "Sentiment Analysis",
    category: "LLM: Basic",
    needsLLM: true,
    code: `-- Analyze sentiment of reviews
let reviews = [
    "This product is amazing! Best purchase ever.",
    "Terrible quality, broke after one day.",
    "It's okay, nothing special but works fine."
]

let sentiments = map reviews with
    ask "Rate the sentiment: positive, negative, or neutral. Reply with ONE word.\\n\\nReview: {it}"

let results = map (zip(reviews, sentiments)) with
    "[" ++ (last(it) or "?") ++ "] " ++ slice((first(it) or ""), 0, 40) ++ "..."

return join results with "\\n"`,
  },
  extract: {
    name: "Structured Extraction",
    category: "LLM: Basic",
    needsLLM: true,
    code: `-- Extract structured data with typed ask
let text = "The meeting attendees were Alice from Engineering, Bob from Marketing, and Charlie from Design. They discussed the Q3 roadmap."

let people = ask "List only the people's names. Return a JSON array of strings.\\n\\n{text}" as List<String> with retries: 2 fallback ["unknown"]

return "Found " ++ show(length(people)) ++ " people: " ++ join people with ", "`,
  },

  // ─── LLM: Advanced ──────────────────────────────────────────────────────
  summarize: {
    name: "Summarization",
    category: "LLM: Advanced",
    needsLLM: true,
    code: `-- Summarize text with an LLM
let article = "Domain-specific languages (DSLs) for LLM orchestration address safety challenges in AI systems. Unlike general-purpose languages like Python, these DSLs enforce resource limits at the language level, preventing runaway token usage and unbounded loops. By embracing functional programming concepts like immutability and pure functions, DSLs make LLM pipelines easy to test and debug."

return ask "Summarize this in exactly one sentence:\\n\\n{article}"`,
  },
  multistep: {
    name: "Multi-Step Analysis",
    category: "LLM: Advanced",
    needsLLM: true,
    code: `-- Chain multiple LLM calls (map-reduce pattern)
let paragraphs = [
    "AI safety is crucial as systems become more capable.",
    "Functional programming reduces bugs through immutability.",
    "DSLs can enforce constraints that general languages cannot."
]

let summaries = map paragraphs with
    ask "Summarize in 5 words or less: {it}"

let combined = join summaries with "; "

return ask "Synthesize these points into one insight: {combined}"`,
  },
  channel: {
    name: "Model Channels",
    category: "LLM: Advanced",
    needsLLM: true,
    code: `-- Route to different models via channels
let question = "Explain quantum entanglement simply."

-- Channel routes to different models (fast=small, smart=large)
let quick = ask question via fast
let detailed = ask question via smart

return "Quick answer: " ++ quick ++ "\\n\\nDetailed: " ++ detailed`,
  },
};

interface Props {
  initialExample?: string;
}

export default function Editor({ initialExample = "hello" }: Props) {
  const editorRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<EditorView | null>(null);
  const [output, setOutput] = useState<string>("");
  const [isError, setIsError] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [selectedExample, setSelectedExample] = useState(initialExample);
  const [hoistModule, setHoistModule] = useState<any>(null);
  const [llmMode, setLlmMode] = useState<LLMMode>("golden");
  const [wllamaStatus, setWllamaStatus] = useState<string>("");
  const wllamaRef = useRef<any>(null);

  // Load Hoist WASM module on mount
  useEffect(() => {
    import("../wasm/hoist_wasm.js").then((mod) => {
      setHoistModule(mod);
    });
  }, []);

  // Load wllama when mode switches to "wllama"
  useEffect(() => {
    if (llmMode !== "wllama" || wllamaRef.current) return;

    let cancelled = false;
    (async () => {
      try {
        setWllamaStatus("Loading wllama engine...");
        const { Wllama } = await import("@wllama/wllama");
        if (cancelled) return;

        const wllama = new Wllama({
          "single-thread/wllama.wasm": "/wllama/single-thread.wasm",
          "multi-thread/wllama.wasm": "/wllama/multi-thread.wasm",
        });

        setWllamaStatus("Downloading SmolLM2-135M (~105MB)...");
        await wllama.loadModelFromHF(
          "bartowski/SmolLM2-135M-Instruct-GGUF",
          "SmolLM2-135M-Instruct-Q4_K_M.gguf",
          {
            progressCallback: ({ loaded, total }: { loaded: number; total: number }) => {
              if (cancelled) return;
              const pct = total > 0 ? Math.round((loaded / total) * 100) : 0;
              setWllamaStatus(`Downloading model... ${pct}%`);
            },
          }
        );
        if (cancelled) return;

        wllamaRef.current = wllama;
        setWllamaStatus("SmolLM2-135M ready");
      } catch (err) {
        setWllamaStatus(`Failed: ${err}`);
        setLlmMode("golden"); // fall back
      }
    })();

    return () => { cancelled = true; };
  }, [llmMode]);

  // Initialize CodeMirror
  useEffect(() => {
    if (!editorRef.current) return;

    const state = EditorState.create({
      doc: EXAMPLES[selectedExample]?.code || "",
      extensions: [
        lineNumbers(),
        highlightActiveLine(),
        history(),
        keymap.of([...defaultKeymap, ...historyKeymap]),
        oneDark,
        EditorView.theme({
          "&": { height: "100%", fontSize: "14px" },
          ".cm-scroller": { overflow: "auto", fontFamily: "var(--font-mono)" },
          ".cm-content": { padding: "1rem 0" },
        }),
      ],
    });

    const view = new EditorView({
      state,
      parent: editorRef.current,
    });

    viewRef.current = view;

    return () => view.destroy();
  }, []);

  // Update editor content when example changes
  useEffect(() => {
    if (viewRef.current && EXAMPLES[selectedExample]) {
      const view = viewRef.current;
      view.dispatch({
        changes: {
          from: 0,
          to: view.state.doc.length,
          insert: EXAMPLES[selectedExample].code,
        },
      });
      setOutput("");
      setIsError(false);
    }
  }, [selectedExample]);

  const runCode = async () => {
    if (!viewRef.current || !hoistModule) return;

    const source = viewRef.current.state.doc.toString();
    const needsLLM = EXAMPLES[selectedExample]?.needsLLM;
    setIsRunning(true);
    setIsError(false);

    try {
      if (needsLLM && llmMode === "wllama" && wllamaRef.current) {
        // ── wllama mode: two-pass ──
        // Pass 1: collect all ask prompts
        const prompts: { prompt: string; channel: string | null }[] = [];
        const collectRuntime = new hoistModule.HoistRuntime();
        collectRuntime.setAskHandler((prompt: string, channel: string | null) => {
          prompts.push({ prompt, channel });
          // Return a placeholder so execution completes
          return goldenAskHandler(prompt, channel);
        });
        collectRuntime.run(source);

        // Generate real responses from wllama
        setOutput("Generating LLM responses...");
        const cache = new Map<string, string>();
        for (let i = 0; i < prompts.length; i++) {
          const { prompt, channel } = prompts[i];
          const key = `${channel || ""}:${prompt}`;
          if (!cache.has(key)) {
            setOutput(`Generating response ${i + 1}/${prompts.length}...`);
            const response = await wllamaRef.current.createChatCompletion(
              [{ role: "user", content: prompt }],
              { nPredict: 128, sampling: { temp: 0.2, top_k: 20, top_p: 0.9 } }
            );
            cache.set(key, response.trim());
          }
        }

        // Pass 2: replay with real responses
        let idx = 0;
        const replayRuntime = new hoistModule.HoistRuntime();
        replayRuntime.setAskHandler((prompt: string, channel: string | null) => {
          const key = `${channel || ""}:${prompt}`;
          if (cache.has(key)) return cache.get(key)!;
          // Fallback for any prompts that differ between passes
          return prompts[idx++]
            ? cache.get(`${prompts[idx - 1].channel || ""}:${prompts[idx - 1].prompt}`) || "(no response)"
            : "(no response)";
        });
        const result = replayRuntime.run(source);
        setOutput(typeof result === "string" ? result : JSON.stringify(result, null, 2));
      } else {
        // ── golden or prompt mode (synchronous) ──
        const runtime = new hoistModule.HoistRuntime();

        if (needsLLM) {
          if (llmMode === "golden") {
            runtime.setAskHandler(goldenAskHandler);
          } else {
            // prompt mode: window.prompt()
            runtime.setAskHandler((prompt: string, _channel: string | null) => {
              const response = window.prompt(
                `LLM Ask:\n\n${prompt}\n\n(Enter a mock response)`
              );
              return response || "(no response)";
            });
          }
        }

        const result = runtime.run(source);
        setOutput(typeof result === "string" ? result : JSON.stringify(result, null, 2));
      }
    } catch (err: unknown) {
      setIsError(true);
      setOutput(String(err));
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div class="editor-container">
      <div class="toolbar">
        <select
          value={selectedExample}
          onChange={(e) => setSelectedExample((e.target as HTMLSelectElement).value)}
        >
          {Object.entries(
            Object.entries(EXAMPLES).reduce((acc, [key, ex]) => {
              if (!acc[ex.category]) acc[ex.category] = [];
              acc[ex.category].push([key, ex]);
              return acc;
            }, {} as Record<string, [string, typeof EXAMPLES[string]][]>)
          ).map(([category, items]) => (
            <optgroup key={category} label={category}>
              {items.map(([key, { name }]) => (
                <option key={key} value={key}>
                  {name}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
        {EXAMPLES[selectedExample]?.needsLLM && (
          <select
            class="llm-mode-select"
            value={llmMode}
            onChange={(e) => setLlmMode((e.target as HTMLSelectElement).value as LLMMode)}
          >
            <option value="golden">Simulated LLM</option>
            <option value="wllama">Real LLM (SmolLM2-135M)</option>
            <option value="prompt">Manual (prompt)</option>
          </select>
        )}
        <button onClick={runCode} disabled={isRunning || !hoistModule}>
          {isRunning ? "Running..." : hoistModule ? "Run ▶" : "Loading WASM..."}
        </button>
      </div>
      {wllamaStatus && llmMode === "wllama" && (
        <div class="wllama-status">{wllamaStatus}</div>
      )}

      <div class="editor-wrapper">
        <div class="pane">
          <div class="pane-header">Source</div>
          <div class="editor-content" ref={editorRef}></div>
        </div>

        <div class="pane">
          <div class="pane-header">Output</div>
          <pre class={`output-content ${isError ? "error" : ""}`}>
            {output || "(click Run to execute)"}
          </pre>
        </div>
      </div>
    </div>
  );
}
