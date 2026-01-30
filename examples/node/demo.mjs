#!/usr/bin/env node
/**
 * Hoist Node.js demo — runs example programs via the CLI binary,
 * using Ollama as the LLM backend.
 *
 * Usage:
 *     cd <repo-root>
 *     node examples/node/demo.mjs
 *
 * Prerequisites:
 *     - cargo build --release -p hoist-cli   (produces target/release/hoist)
 *     - Ollama running locally for programs 03–06
 *
 * Zero npm dependencies — uses Node 22 built-in fetch + child_process.
 */

import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, "..", "..");
const PROGRAMS_DIR = join(REPO_ROOT, "examples", "programs");
const DATA_DIR = join(REPO_ROOT, "examples", "data");
const HOIST_BIN = join(REPO_ROOT, "target", "release", "hoist");

const OLLAMA_URL = "http://localhost:11434/api/generate";
const CHANNEL_MODELS = {
  fast: "gemma3:1b",
  smart: "deepseek-r1:7b",
  default: "qwen3:4b",
};

// ── Helpers ────────────────────────────────────────────────────────────

function banner(msg) {
  const width = Math.max(msg.length + 4, 60);
  console.log("\n" + "=".repeat(width));
  console.log(`  ${msg}`);
  console.log("=".repeat(width));
}

function stripThinkTags(text) {
  return text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
}

async function ollamaAsk(prompt, channel) {
  const model = CHANNEL_MODELS[channel || "default"] || CHANNEL_MODELS.default;
  const actualPrompt = model.startsWith("qwen3") ? "/no_think " + prompt : prompt;

  const resp = await fetch(OLLAMA_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, prompt: actualPrompt, stream: false }),
    signal: AbortSignal.timeout(120_000),
  });

  if (!resp.ok) {
    throw new Error(`Ollama HTTP ${resp.status}: ${await resp.text()}`);
  }
  const body = await resp.json();
  return stripThinkTags(body.response || "");
}

// ── Run a Hoist program via CLI subprocess ─────────────────────────────

function runHoist(programPath, { context } = {}) {
  return new Promise((resolve, reject) => {
    const args = [programPath];
    if (context) {
      args.push("--context-file", context);
    }

    const child = spawn(HOIST_BIN, args, {
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderrBuf = "";
    let pendingPrompt = null;

    child.stdout.on("data", (d) => (stdout += d.toString()));

    child.stderr.on("data", async (data) => {
      stderrBuf += data.toString();

      // Process complete lines from stderr
      while (stderrBuf.includes("\n")) {
        const nlIdx = stderrBuf.indexOf("\n");
        const line = stderrBuf.slice(0, nlIdx);
        stderrBuf = stderrBuf.slice(nlIdx + 1);

        // Detect [ask] or [ask via <channel>] prompt lines
        const askMatch = line.match(/^\[ask(?:\s+via\s+(\S+))?\]\s+(.*)/);
        if (askMatch) {
          const [, channel, text] = askMatch;
          if (text === "Enter response (end with empty line):") {
            // This is the "enter response" line — send the LLM response
            if (pendingPrompt !== null) {
              try {
                const answer = await ollamaAsk(
                  pendingPrompt.text,
                  pendingPrompt.channel
                );
                child.stdin.write(answer + "\n\n");
              } catch (err) {
                child.stdin.write(`Error: ${err.message}\n\n`);
              }
              pendingPrompt = null;
            }
          } else {
            // Accumulate multi-line prompt
            if (pendingPrompt !== null) {
              pendingPrompt.text += "\n" + text;
            } else {
              pendingPrompt = { text, channel: channel || null };
            }
          }
        } else if (pendingPrompt !== null && !line.startsWith("[ask]")) {
          // Continuation line of a multi-line prompt
          pendingPrompt.text += "\n" + line;
        }
      }
    });

    child.on("close", (code) => {
      if (code === 0) {
        resolve(stdout.trim());
      } else {
        reject(new Error(`hoist exited with code ${code}\nstderr: ${stderrBuf}`));
      }
    });

    child.on("error", reject);
  });
}

// ── Main ───────────────────────────────────────────────────────────────

async function main() {
  banner("Hoist Node.js Demo");

  const contextFile = join(DATA_DIR, "sample_article.txt");

  const programs = [
    { file: "01_hello.hoist", llm: false },
    { file: "02_text_pipeline.hoist", llm: false },
    { file: "03_summarize.hoist", llm: true, context: contextFile },
    { file: "04_multi_step.hoist", llm: true, context: contextFile },
    { file: "05_classify.hoist", llm: true },
    { file: "06_extract_structured.hoist", llm: true },
  ];

  for (const prog of programs) {
    const path = join(PROGRAMS_DIR, prog.file);
    const source = readFileSync(path, "utf-8");

    banner(`Running ${prog.file}  (${prog.llm ? "LLM via Ollama" : "pure — no LLM"})`);
    console.log(`Source:\n${source.trim().replace(/^/gm, "  ")}\n`);

    try {
      const result = await runHoist(path, {
        context: prog.context,
      });
      console.log(`Result: ${result}`);
    } catch (err) {
      console.log(`Error: ${err.message}`);
      if (prog.llm) {
        console.log("(Is Ollama running?  Start it with: ollama serve)");
      }
    }
  }

  banner("Demo complete");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
