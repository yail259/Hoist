#!/usr/bin/env python3
"""
Hoist Python demo — builds native PyO3 bindings and runs example programs
with Ollama as the LLM backend.

This script bootstraps itself:
  1. Creates a venv at .venv/ (via uv or python -m venv)
  2. Installs maturin into the venv
  3. Re-execs itself inside the venv if needed
  4. Builds hoist-python bindings with maturin develop
  5. Runs all example programs

Usage:
    cd <repo-root>
    python3 examples/python/demo.py

Prerequisites:
    - Rust toolchain (rustc, cargo)
    - uv (recommended) or python3 with venv module
    - Ollama running locally (ollama serve) for programs 03–06
"""

import os
import sys
import subprocess
import textwrap
import shutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROGRAMS_DIR = os.path.join(REPO_ROOT, "examples", "programs")
DATA_DIR = os.path.join(REPO_ROOT, "examples", "data")
VENV_DIR = os.path.join(REPO_ROOT, ".venv")

# ── Helpers ──────────────────────────────────────────────────────────────

def banner(msg: str) -> None:
    width = max(len(msg) + 4, 60)
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()


def venv_python() -> str:
    return os.path.join(VENV_DIR, "bin", "python3")


def in_venv() -> bool:
    """Check if we're running inside our project venv."""
    return os.path.realpath(sys.prefix) == os.path.realpath(VENV_DIR)


# ── Step 0: bootstrap venv + re-exec ────────────────────────────────────

def bootstrap_venv() -> None:
    """Create venv with maturin, then re-exec this script inside it."""
    if in_venv():
        return  # already inside venv

    banner("Bootstrapping virtual environment")

    uv = shutil.which("uv")

    if not os.path.isfile(venv_python()):
        if uv:
            print(f"Creating venv at {VENV_DIR} (via uv) ...")
            subprocess.run([uv, "venv", VENV_DIR, "--python", "3.12"],
                           check=True)
        else:
            print(f"Creating venv at {VENV_DIR} (via python -m venv) ...")
            subprocess.run([sys.executable, "-m", "venv", VENV_DIR],
                           check=True)

    # Install maturin into the venv
    print("Installing maturin ...")
    if uv:
        subprocess.run([uv, "pip", "install", "maturin", "--python",
                        venv_python()], check=True)
    else:
        subprocess.run([venv_python(), "-m", "pip", "install", "maturin"],
                       check=True)

    # Re-exec this script inside the venv
    print(f"\nRe-launching inside venv: {venv_python()}")
    os.execv(venv_python(), [venv_python(), __file__] + sys.argv[1:])


# ── Step 1: build bindings ──────────────────────────────────────────────

def venv_bin(name: str) -> str:
    return os.path.join(VENV_DIR, "bin", name)


def build_bindings() -> None:
    banner("Building Hoist Python bindings (maturin develop)")

    subprocess.run(
        [venv_bin("maturin"), "develop", "--release"],
        cwd=os.path.join(REPO_ROOT, "hoist-python"),
        check=True,
    )
    print("Build complete.\n")


# ── Step 2: run programs ────────────────────────────────────────────────

def run_pure_programs(hoist_mod):
    """Run programs 01–02 that don't need an LLM."""
    for name in ["01_hello.hoist", "02_text_pipeline.hoist"]:
        path = os.path.join(PROGRAMS_DIR, name)
        source = read_file(path)
        banner(f"Running {name}  (pure — no LLM)")
        print(f"Source:\n{textwrap.indent(source.strip(), '  ')}\n")
        result = hoist_mod.run(source)
        print(f"Result: {result}")


def run_llm_programs(hoist_mod):
    """Run programs 03–06 that need Ollama."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "python"))
    from ollama_handler import ollama_ask

    context = read_file(os.path.join(DATA_DIR, "sample_article.txt"))

    programs = [
        ("03_summarize.hoist", True),
        ("04_multi_step.hoist", True),
        ("05_classify.hoist", False),
        ("06_extract_structured.hoist", False),
    ]

    for name, needs_context in programs:
        path = os.path.join(PROGRAMS_DIR, name)
        source = read_file(path)
        banner(f"Running {name}  (LLM via Ollama)")
        print(f"Source:\n{textwrap.indent(source.strip(), '  ')}\n")

        rt = hoist_mod.HoistRuntime(ask_handler=ollama_ask)
        if needs_context:
            rt.set_context(context)
        result = rt.run(source)
        print(f"Result: {result}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    banner("Hoist Python Demo")

    bootstrap_venv()   # no-op if already in venv; re-execs otherwise
    build_bindings()

    import hoist  # available after maturin develop
    print(f"Imported hoist module: {hoist}")

    run_pure_programs(hoist)

    # LLM programs are optional — skip gracefully if Ollama isn't running
    try:
        run_llm_programs(hoist)
    except RuntimeError as exc:
        print(f"\nSkipping LLM programs: {exc}")
        print("Start Ollama with:  ollama serve")

    banner("Demo complete")


if __name__ == "__main__":
    main()
