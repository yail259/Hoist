#!/usr/bin/env bash
# Hoist demo launcher — builds CLI, runs pure examples, then Python + Node demos.
#
# Usage:
#     ./examples/run_demo.sh          # run everything
#     ./examples/run_demo.sh pure     # only pure (no-LLM) examples
#     ./examples/run_demo.sh python   # only Python demo
#     ./examples/run_demo.sh node     # only Node.js demo

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PROGRAMS="$REPO_ROOT/examples/programs"
DATA="$REPO_ROOT/examples/data"
HOIST="$REPO_ROOT/target/release/hoist"

banner() { printf '\n%s\n  %s\n%s\n' "$(printf '=%.0s' {1..60})" "$1" "$(printf '=%.0s' {1..60})"; }

# ── Build CLI ──────────────────────────────────────────────────────────

banner "Building hoist CLI (release)"
cargo build --release -p hoist-cli
echo "Binary: $HOIST"

# ── Check Ollama ──────────────────────────────────────────────────────

check_ollama() {
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is running."
        return 0
    else
        echo "WARNING: Ollama not detected at localhost:11434"
        echo "LLM examples will be skipped.  Start Ollama with: ollama serve"
        return 1
    fi
}

# ── Pure examples (no LLM) ────────────────────────────────────────────

run_pure() {
    banner "Pure examples (no LLM needed)"

    for prog in "$PROGRAMS"/01_hello.hoist "$PROGRAMS"/02_text_pipeline.hoist; do
        name="$(basename "$prog")"
        echo ""
        echo "--- $name ---"
        "$HOIST" "$prog"
    done
}

# ── Python demo ───────────────────────────────────────────────────────

run_python() {
    banner "Python demo (PyO3 bindings + Ollama)"
    python3 "$REPO_ROOT/examples/python/demo.py"
}

# ── Node.js demo ─────────────────────────────────────────────────────

run_node() {
    banner "Node.js demo (CLI subprocess + Ollama)"
    node "$REPO_ROOT/examples/node/demo.mjs"
}

# ── Dispatch ─────────────────────────────────────────────────────────

MODE="${1:-all}"

case "$MODE" in
    pure)
        run_pure
        ;;
    python)
        check_ollama || true
        run_python
        ;;
    node)
        check_ollama || true
        run_node
        ;;
    all)
        run_pure
        if check_ollama; then
            run_python
            run_node
        else
            echo ""
            echo "Skipping LLM demos (Ollama not running)."
            echo "Pure examples completed successfully above."
        fi
        ;;
    *)
        echo "Usage: $0 [pure|python|node|all]"
        exit 1
        ;;
esac

banner "Done"
