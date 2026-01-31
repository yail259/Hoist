# Hoist

**A functional DSL for safe LLM orchestration.**

---

## What is Hoist?

Hoist is a pure functional domain-specific language designed to orchestrate LLM calls safely. Resource limits -- max LLM calls, string sizes, collection sizes, and execution steps -- are enforced at the language level, so programs cannot loop forever, access the filesystem or network, or make uncontrolled LLM calls. All LLM I/O goes through a host-provided handler, giving the host full control and auditability.

Hoist is written in Rust, with bindings for Python (PyO3), Node.js (NAPI-rs), and the browser (WebAssembly).

## Quick Example

```hoist
let chunks = split context by "\n\n"
let summaries = map chunks with
    ask "Summarize: {it}" with retries: 2 fallback "(failed)"
return join summaries with "\n"
```

## Key Features

- **Enforced resource limits** -- `max_ask_calls`, `max_string_size`, `max_collection_size`, `max_steps` are all bounded and configurable
- **Pure functional** -- no mutable state, no side effects, no filesystem or network access
- **Typed LLM responses** -- `ask "..." as List<String>` with retries and fallback values
- **Pipeline operator** -- `|>` for clean, composable data flow
- **Runs sandboxed** in Python, Node.js, or the browser (WASM)
- **All LLM I/O goes through your handler** -- fully auditable, swap providers without changing programs

## Installation

**Python** (requires Rust toolchain + [maturin](https://github.com/PyO3/maturin)):

```bash
cd hoist-python
maturin develop --release
```

**CLI:**

```bash
cargo install --path hoist-cli
```

**Browser:** WebAssembly via the [playground](#try-it).

## Usage

### Python

```python
import hoist

def my_handler(prompt, channel):
    return openai.complete(prompt)

rt = hoist.HoistRuntime(ask_handler=my_handler)
rt.set_context("Your text here")
result = rt.run('let chunks = split context by "\\n\\n"\nreturn show(length(chunks))')
print(result)
```

### CLI

```bash
# Run a program file
hoist program.hoist --context "Hello world"

# Evaluate inline code
hoist --eval 'return upper("hello")'

# Read from stdin
cat program.hoist | hoist --context-file data.txt

# Parse-check only (no execution)
hoist program.hoist --check
```

The CLI reads LLM responses interactively from stdin. To use a real LLM backend, use the Python or Node.js bindings with a custom ask handler.

## Try It

The playground runs entirely in the browser via WebAssembly. No signup, no server, no data leaves your machine. See the `playground/` directory to build and run it locally.

## Language Overview

**Bindings and control flow:**

```
let x = 42
return x + 1
if x > 0 then "positive" else "non-positive"
match value with "a" -> 1, "b" -> 2, _ -> 0
```

**Collections:** lists, records, tuples.

**Higher-order functions:** `map`, `filter`, `fold`, `take`, `drop`.

**String functions:** `split`, `join`, `lines`, `words`, `upper`, `lower`, `trim`, `contains`, `replace`, `find_all`, `starts_with`, `length`.

**LLM calls:**

```
ask "prompt"
ask "prompt" via channel
ask "prompt" as List<String>
ask "prompt" with retries: 3 fallback "default"
```

**Pipeline operator:**

```
data |> filter where length(it) > 5 |> map with upper(it)
```

See [hoist-spec.md](hoist-spec.md) for the full language specification.

## Project Structure

| Crate | Path | Description |
|-------|------|-------------|
| `hoist-core` | `hoist-core/` | Parser, interpreter, type system, resource limits |
| `hoist-cli` | `hoist-cli/` | Command-line tool (`hoist` binary) |
| `hoist-python` | `hoist-python/` | Python bindings via PyO3 |
| `hoist-node` | `hoist-node/` | Node.js bindings via NAPI-rs |
| `hoist-wasm` | `hoist-wasm/` | WebAssembly bindings via wasm-bindgen |
| playground | `playground/` | Browser-based playground (Astro + WASM) |
| examples | `examples/` | Demo programs, Python/Node scripts, sample data |

## Running Examples

```bash
# Run everything (pure examples + LLM demos if Ollama is available)
./examples/run_demo.sh

# Run only pure (no-LLM) examples
./examples/run_demo.sh pure

# Run only the Python demo
./examples/run_demo.sh python

# Run only the Node.js demo
./examples/run_demo.sh node
```

LLM examples require [Ollama](https://ollama.ai/) running locally (`ollama serve`). Pure examples (programs 01-02) work without any LLM backend.

## Running Tests

```bash
cargo test --workspace
```

## License

MIT
