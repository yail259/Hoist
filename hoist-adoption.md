# Hoist Adoption Guide

A practical guide for developer experience, distribution, and integrating Hoist into production systems.

---

## Table of Contents

1. [Distribution Model](#1-distribution-model)
2. [LLM Prompting](#2-llm-prompting)
3. [SDK Design](#3-sdk-design)
4. [Error Handling & Debugging](#4-error-handling--debugging)
5. [Observability](#5-observability)
6. [Testing Strategies for Adopters](#6-testing-strategies-for-adopters)
7. [Migration Path](#7-migration-path)
8. [Security Considerations for Adopters](#8-security-considerations-for-adopters)
9. [Performance Tuning](#9-performance-tuning)
10. [Common Pitfalls](#10-common-pitfalls)
11. [Example Integrations](#11-example-integrations)
12. [Versioning & Compatibility](#12-versioning--compatibility)
13. [Support & Community](#13-support--community)

---

## 1. Distribution Model

### How It Ships

Hoist is implemented in Rust and distributed as native bindings for each target language. Users install it like any other package—no separate runtime, no Rust toolchain required.

```
┌─────────────────────────────────────────────────────────┐
│                   Rust Core (hoist-core)                │
│         Lexer → Parser → Type Checker → Interpreter     │
└─────────────────────────┬───────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   ┌─────────┐      ┌──────────┐      ┌─────────┐
   │  PyO3   │      │ napi-rs  │      │  WASM   │
   └────┬────┘      └────┬─────┘      └────┬────┘
        │                │                 │
        ▼                ▼                 ▼
   ┌─────────┐      ┌──────────┐      ┌─────────┐
   │   pip   │      │   npm    │      │ <script>│
   │ install │      │ install  │      │   tag   │
   └─────────┘      └──────────┘      └─────────┘
```

### Python Installation

```bash
pip install hoist
```

The wheel contains a compiled native library. pip automatically selects the correct wheel for the user's platform:

| Platform | Wheel Suffix |
|----------|--------------|
| Linux x64 (glibc) | `manylinux_2_17_x86_64` |
| Linux ARM64 | `manylinux_2_17_aarch64` |
| macOS x64 | `macosx_10_12_x86_64` |
| macOS ARM64 (M1/M2) | `macosx_11_0_arm64` |
| Windows x64 | `win_amd64` |

Users just `import hoist`—no configuration needed.

### Node.js Installation

```bash
npm install @anthropic/hoist
```

The package uses optional dependencies to ship platform-specific binaries:

```json
{
  "name": "@anthropic/hoist",
  "optionalDependencies": {
    "@anthropic/hoist-linux-x64-gnu": "0.1.0",
    "@anthropic/hoist-linux-arm64-gnu": "0.1.0",
    "@anthropic/hoist-darwin-x64": "0.1.0",
    "@anthropic/hoist-darwin-arm64": "0.1.0",
    "@anthropic/hoist-win32-x64-msvc": "0.1.0"
  }
}
```

npm installs only the matching platform package. The main package detects and loads the correct binary at runtime.

### Browser / Edge (WASM)

```html
<script type="module">
  import { createRuntime } from 'https://unpkg.com/@anthropic/hoist-wasm';

  const runtime = await createRuntime();
  const result = await runtime.eval('return 1 + 2', {});
</script>
```

WASM builds work anywhere with a modern JavaScript runtime—browsers, Deno, Cloudflare Workers, etc.

### Fallback Strategy

For exotic platforms without prebuilt binaries, the package can fall back to WASM:

```python
# hoist/__init__.py
try:
    from hoist._native import HoistRuntime
except ImportError:
    from hoist._wasm import HoistRuntime
    import warnings
    warnings.warn(
        "Native Hoist binary not available for this platform. "
        "Falling back to WASM (slower). "
        "Please report this at github.com/anthropic/hoist/issues"
    )
```

---

## 2. LLM Prompting

### The Problem

Hoist is useless if LLMs can't write valid Hoist code. The quality of generated code depends entirely on how you prompt the LLM.

### Shipped Prompt Assets

The package includes ready-to-use prompt materials:

```
hoist/
└── prompts/
    ├── system_prompt.txt       # Full system prompt (~2000 tokens)
    ├── reference_card.txt      # Condensed syntax reference (~500 tokens)
    ├── few_shot_examples.txt   # Task → Hoist examples (~1000 tokens)
    └── constraints.txt         # Security/safety constraints (~200 tokens)
```

### System Prompt (Full)

```markdown
You are a code generator that writes Hoist programs. Hoist is a safe,
functional language for LLM orchestration.

## Syntax Overview

### Variables and Returns
```hoist
let x = 42
let name = "Alice"
return x + 1
```

### Pipelines
Chain operations left-to-right with |>
```hoist
data |> split by "\n" |> filter where len(it) > 0 |> map with upper
```

### Control Flow
```hoist
if condition then value1 else value2
match x { 1 -> "one", 2 -> "two", _ -> "other" }
```

### Lambdas
```hoist
x -> x * 2              // Single parameter
(a, b) -> a + b         // Multiple parameters
filter where it > 0     // Implicit 'it' for single-param
```

## Available Functions

### String Functions
- len(s) -> Int
- upper(s), lower(s), trim(s) -> String
- split(s, delim) -> List[String]
- join(list, delim) -> String
- contains(s, substr) -> Bool
- replace(s, old, new) -> String
- substr(s, start, end) -> String
- matches(s, regex) -> List[String]

### List Functions
- len(list) -> Int
- map(list, fn) -> List
- filter(list, predicate) -> List
- reduce(list, initial, fn) -> Value
- first(list), last(list) -> Value
- take(list, n), drop(list, n) -> List
- sort(list), unique(list) -> List
- flatten(list) -> List
- range(start, end) -> List[Int]

### LLM Integration (AI-Friendly Features)
- ask(prompt) -> String                    // Basic LLM call
- ask(prompt) as Type                      // Typed output (auto-parsed)
- ask(prompt) via channel                  // Route to specific model/config
- ask(prompt) with retries: N              // Auto-retry on failure
- ask(prompt) fallback expr                // Fallback value on failure

Examples:
```hoist
-- Get structured data automatically
let fruits = ask "List 3 fruits" as List<String>

-- Route to different models
let summary = ask "Summarize: {text}" via fast_model
let analysis = ask "Deep analysis: {text}" via smart_model

-- Handle failures gracefully
let result = ask "Complex task" with retries: 3 fallback "default"

-- Combine modifiers
let data = ask "Extract: {text}" as {name: String, age: Int}
    via extractor
    with retries: 2
    fallback {name: "Unknown", age: 0}
```

### Type Conversion
- int(x), float(x), str(x), bool(x)
- parse_json(s), to_json(value)

## Constraints

- No loops (use map/filter/reduce instead)
- No recursion
- No file or network access
- No mutation (all data is immutable)
- Programs must terminate

## Examples

Task: "Count words in each paragraph"
```hoist
let paragraphs = split context by "\n\n"
return map paragraphs with (p -> len(split(p, " ")))
```

Task: "Find paragraphs mentioning 'revenue' and summarize each"
```hoist
let paragraphs = split context by "\n\n"
let relevant = filter paragraphs where contains(lower(it), "revenue")
let summaries = map relevant with (p -> ask("Summarize in one sentence: {p}"))
return join summaries with "\n"
```

Task: "Extract all email addresses"
```hoist
return matches(context, "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}")
```

When writing Hoist code:
1. Use pipelines for data transformation chains
2. Use 'it' for simple single-parameter lambdas
3. Use 'ask' when you need semantic understanding
4. Always include a 'return' statement
```

### Reference Card (Condensed)

For context-limited situations, use the condensed reference:

```markdown
# Hoist Quick Reference

## Syntax
let x = value           // Variable binding
return expr             // Return value (required)
a |> fn                 // Pipeline: fn(a)
a |> fn with b          // Pipeline: fn(a, b)
x -> expr               // Lambda
it                      // Implicit lambda param

## Strings
len upper lower trim split join contains replace substr matches

## Lists
len map filter reduce first last take drop sort unique flatten range

## LLM
ask(prompt)             // Returns string response

## Types
int float str bool parse_json to_json

## Control
if cond then a else b
match x { pattern -> result, _ -> default }
```

### Accessing Prompts Programmatically

```python
import hoist

# Get the full system prompt
system_prompt = hoist.prompts.system_prompt()

# Get just the reference card (for appending to existing prompts)
reference = hoist.prompts.reference_card()

# Get few-shot examples
examples = hoist.prompts.few_shot_examples()

# Build a custom prompt
custom_prompt = hoist.prompts.build(
    include_syntax=True,
    include_functions=True,
    include_examples=True,
    include_constraints=True,
    max_tokens=1500  # Truncate if needed
)
```

```javascript
import { prompts } from '@anthropic/hoist';

const systemPrompt = prompts.systemPrompt();
const reference = prompts.referenceCard();
```

### Prompt Engineering Tips

**Do:**
- Include the full system prompt when generating Hoist code
- Provide task-specific examples when possible
- Tell the LLM what variables are available in context
- Ask for explanation alongside code (helps catch errors)

**Don't:**
- Assume the LLM knows Hoist without prompting
- Use generic "write code" prompts
- Forget to specify available context variables

**Example Integration:**

```python
import hoist
from openai import OpenAI

client = OpenAI()

def generate_hoist_program(task: str, available_vars: list[str]) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": hoist.prompts.system_prompt()
            },
            {
                "role": "user",
                "content": f"""Write a Hoist program for this task:

Task: {task}

Available variables: {', '.join(available_vars)}

Return only the Hoist code, no explanation."""
            }
        ]
    )
    return response.choices[0].message.content
```

---

## 3. SDK Design

### Python API

```python
import hoist

# Create a runtime with an LLM callback
runtime = hoist.Runtime(
    ask=lambda prompt: call_my_llm(prompt),  # Your LLM function
    max_iterations=10_000,                    # Optional limits
    max_recursion=100,
    timeout_ms=30_000
)

# Execute a program with context
result = runtime.eval(
    program="""
        let chunks = split context by "\\n\\n"
        let relevant = filter chunks where contains(it, query)
        return map relevant with (c -> ask("Summarize: {c}"))
    """,
    context={
        "context": long_document,
        "query": "revenue"
    }
)

# Result is a Python object (list, dict, str, int, etc.)
for summary in result:
    print(summary)
```

### Node.js API

```javascript
import { Runtime } from '@anthropic/hoist';

const runtime = new Runtime({
    ask: async (prompt) => await callMyLLM(prompt),
    maxIterations: 10_000,
    maxRecursion: 100,
    timeoutMs: 30_000
});

const result = await runtime.eval(
    `let chunks = split context by "\\n\\n"
     return map chunks with (c -> ask("Summarize: {c}"))`,
    { context: longDocument }
);
```

### Type Definitions (TypeScript)

```typescript
interface RuntimeOptions {
    // Default ask handler (used when no channel specified)
    ask?: (request: AskRequest) => Promise<AskResponse>;

    // Named channels for routing (AI-friendly feature)
    channels?: Record<string, (request: AskRequest) => Promise<AskResponse>>;

    maxIterations?: number;
    maxRecursion?: number;
    timeoutMs?: number;
    onLog?: (level: LogLevel, message: string) => void;
}

interface AskRequest {
    prompt: string;
    channel?: string;          // Which channel was requested
    expectedType?: TypeSpec;   // For typed output parsing
    retryCount: number;        // Which attempt (0 = first)
}

interface AskResponse {
    raw: string;               // Raw LLM response
    parsed?: Value;            // Pre-parsed value (optional)
    metadata?: {
        model?: string;
        tokensUsed?: number;
        latencyMs?: number;
    };
}

interface Runtime {
    eval(program: string, context?: Record<string, Value>): Promise<Value>;
    parse(program: string): ParseResult;
    typecheck(program: string): TypecheckResult;
}

type Value =
    | null
    | boolean
    | number
    | string
    | Value[]
    | Record<string, Value>;

type LogLevel = 'debug' | 'info' | 'warn' | 'error';
```

### Channel Configuration (AI-Friendly Routing)

Configure different LLM backends for different tasks:

```python
import hoist
from openai import OpenAI
from anthropic import Anthropic

openai = OpenAI()
anthropic = Anthropic()

runtime = hoist.Runtime(
    channels={
        # Fast, cheap model for simple tasks
        "fast": lambda req: openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": req.prompt}],
            max_tokens=200
        ).choices[0].message.content,

        # Smart model for complex reasoning
        "smart": lambda req: anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": req.prompt}],
            max_tokens=2000
        ).content[0].text,

        # Specialized extractor with system prompt
        "extractor": lambda req: openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Extract structured data precisely."},
                {"role": "user", "content": req.prompt}
            ],
            temperature=0
        ).choices[0].message.content,

        # Default fallback
        "default": lambda req: openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": req.prompt}]
        ).choices[0].message.content,
    }
)

# Now Hoist programs can route to different models:
program = """
let chunks = split context by "\\n\\n"
let summaries = map chunks with ask "Brief summary: {it}" via fast
let analysis = ask "Deep analysis of: {join(summaries, ' ')}" via smart
return analysis
"""
```

**Channel use cases:**
- **Cost optimization**: Use cheaper models for simple tasks
- **Specialization**: Different system prompts per task type
- **Latency**: Fast models for real-time, slow models for batch
- **A/B testing**: Route same channel to different providers
- **Fallback**: Primary model → backup model

### Async Considerations

The `ask` callback is inherently async (LLM API calls). The SDK handles this differently per language:

**Python (sync API with async callback):**
```python
# The runtime handles async internally
runtime = hoist.Runtime(ask=my_sync_or_async_llm_function)
result = runtime.eval(program, context)  # Blocks until complete
```

**Python (async API):**
```python
runtime = hoist.AsyncRuntime(ask=my_async_llm_function)
result = await runtime.eval(program, context)
```

**Node.js (always async):**
```javascript
const result = await runtime.eval(program, context);
```

### Streaming Results

For long-running programs, provide progress callbacks:

```python
runtime = hoist.Runtime(
    ask=my_llm,
    on_progress=lambda event: print(f"Step: {event}")
)

# Events include:
# - AskStarted(prompt)
# - AskCompleted(prompt, response)
# - IterationProgress(current, max)
```

---

## 4. Error Handling & Debugging

### Error Types

Hoist surfaces structured errors that map to specific failure modes:

```python
import hoist
from hoist.errors import (
    ParseError,        # Syntax error in Hoist code
    TypeError,         # Type checking failed
    RuntimeError,      # Execution error
    LimitExceeded,     # Hit iteration/recursion/timeout limit
    HostError,         # The ask callback failed
)

try:
    result = runtime.eval(program, context)
except ParseError as e:
    print(f"Syntax error at line {e.line}, column {e.column}: {e.message}")
    print(e.snippet)  # Shows the problematic code with ^^^ pointer
except TypeError as e:
    print(f"Type error: {e.message}")
    print(f"Expected {e.expected}, got {e.actual}")
except LimitExceeded as e:
    print(f"Program exceeded {e.limit_type}: {e.value} > {e.max}")
except HostError as e:
    print(f"LLM call failed: {e.message}")
    if e.retryable:
        # Could retry the whole program
        pass
except RuntimeError as e:
    print(f"Runtime error: {e.message}")
```

### Error Messages

Hoist prioritizes helpful error messages:

**Bad:**
```
Error: unexpected token
```

**Good:**
```
Parse error at line 3, column 15:

  let result = split context by \n
                              ^^

Expected string literal after 'by', but found identifier '\n'.
Did you mean: split context by "\n"
```

### Debug Mode

Enable verbose logging for development:

```python
runtime = hoist.Runtime(
    ask=my_llm,
    debug=True,
    on_log=lambda level, msg: print(f"[{level}] {msg}")
)
```

Output:
```
[debug] Parsing program (127 bytes)
[debug] Parse successful: 5 statements
[debug] Type checking...
[debug] Type check successful, return type: List[String]
[debug] Executing...
[debug] eval: let chunks = split(context, "\n\n")
[debug] split() -> List[3 items]
[debug] eval: let relevant = filter(chunks, <lambda>)
[debug] filter() -> List[2 items]
[debug] eval: map(relevant, <lambda>)
[info] ask() called with prompt: "Summarize: <paragraph 1>"
[info] ask() returned: "This paragraph discusses..."
[info] ask() called with prompt: "Summarize: <paragraph 2>"
[info] ask() returned: "This paragraph covers..."
[debug] map() -> List[2 items]
[debug] Execution complete in 1.23s
```

### Source Maps

For debugging in production, Hoist tracks source locations:

```python
try:
    result = runtime.eval(program, context)
except RuntimeError as e:
    print(e.stack_trace)
```

```
RuntimeError: Division by zero

  at line 5, column 12:
    let avg = total / count
                    ^

  where:
    total = 100
    count = 0
```

---

## 5. Observability

### Structured Logging

Integrate with your existing logging infrastructure:

```python
import logging
import hoist

logger = logging.getLogger("hoist")

runtime = hoist.Runtime(
    ask=my_llm,
    on_log=lambda level, msg: getattr(logger, level)(msg)
)
```

### Metrics

Expose metrics for monitoring:

```python
runtime = hoist.Runtime(
    ask=my_llm,
    on_metrics=lambda metrics: statsd.gauge("hoist", metrics)
)

# Metrics include:
# - execution_time_ms
# - iterations_used
# - ask_calls_count
# - ask_total_latency_ms
# - memory_peak_bytes
```

### OpenTelemetry Integration

```python
from opentelemetry import trace

tracer = trace.get_tracer("hoist")

def traced_ask(prompt: str) -> str:
    with tracer.start_as_current_span("hoist.ask") as span:
        span.set_attribute("prompt.length", len(prompt))
        response = call_llm(prompt)
        span.set_attribute("response.length", len(response))
        return response

runtime = hoist.Runtime(ask=traced_ask)
```

### Audit Logging

For compliance (SOC2, etc.), log all LLM interactions:

```python
def audited_ask(prompt: str) -> str:
    request_id = uuid.uuid4()

    audit_log.info("llm_request", extra={
        "request_id": str(request_id),
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_length": len(prompt),
        "timestamp": datetime.utcnow().isoformat()
    })

    response = call_llm(prompt)

    audit_log.info("llm_response", extra={
        "request_id": str(request_id),
        "response_length": len(response),
        "timestamp": datetime.utcnow().isoformat()
    })

    return response

runtime = hoist.Runtime(ask=audited_ask)
```

### Typed Output Benefits

The `as Type` modifier dramatically improves reliability:

```python
# WITHOUT typed output - error prone
program_fragile = """
let response = ask "List 3 programming languages"
let languages = parse_json(response)  -- May fail if LLM doesn't return JSON!
return map languages with upper
"""

# WITH typed output - robust
program_robust = """
let languages = ask "List 3 programming languages" as List<String>
return map languages with upper
"""
# Runtime automatically:
# 1. Appends "Respond with a JSON array of strings" to prompt
# 2. Parses the JSON response
# 3. Validates it's actually a list of strings
# 4. Retries with clarification if parsing fails
```

**Supported types:**
- `String` — Raw text (default)
- `Int` — Parsed integer
- `Bool` — true/false
- `List<T>` — JSON array
- `{field: Type, ...}` — JSON object with specific fields

**Complex extraction example:**
```hoist
-- Extract structured data from unstructured text
let person = ask "Extract person info from: {text}" as {
    name: String,
    age: Int,
    skills: List<String>
}

return "Name: {person.name}, Age: {person.age}, Skills: {join(person.skills, ', ')}"
```

---

## 6. Testing Strategies for Adopters

### Unit Testing Hoist Programs

Test your Hoist programs in isolation with mock LLM responses:

```python
import hoist
import pytest

def test_chunking_logic():
    # Mock LLM that returns predictable responses
    responses = iter(["Summary 1", "Summary 2"])
    mock_ask = lambda prompt: next(responses)

    runtime = hoist.Runtime(ask=mock_ask)

    result = runtime.eval(
        """
        let chunks = split context by "\\n\\n"
        return map chunks with (c -> ask("Summarize: {c}"))
        """,
        {"context": "Para 1\\n\\nPara 2"}
    )

    assert result == ["Summary 1", "Summary 2"]

def test_filter_behavior():
    runtime = hoist.Runtime(ask=lambda p: "unused")

    result = runtime.eval(
        """
        let nums = [1, 2, 3, 4, 5]
        return filter nums where it > 3
        """,
        {}
    )

    assert result == [4, 5]
```

### Snapshot Testing

Capture and verify LLM prompts:

```python
def test_prompt_generation(snapshot):
    prompts_sent = []

    def capture_ask(prompt):
        prompts_sent.append(prompt)
        return "mock response"

    runtime = hoist.Runtime(ask=capture_ask)
    runtime.eval(program, {"context": test_document})

    # Verify prompts match expected
    assert prompts_sent == snapshot

### Integration Testing

Test with real LLM in CI (with cost controls):

```python
@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="No API key")
def test_real_llm_integration():
    runtime = hoist.Runtime(
        ask=lambda p: openai.chat(p),
        max_iterations=100  # Limit cost
    )

    result = runtime.eval(
        'return ask("What is 2+2? Reply with just the number.")',
        {}
    )

    assert "4" in result
```

### Property-Based Testing

Test invariants of your Hoist programs:

```python
from hypothesis import given, strategies as st

@given(st.lists(st.integers()))
def test_map_preserves_length(numbers):
    runtime = hoist.Runtime(ask=lambda p: "unused")

    result = runtime.eval(
        f"return map {numbers} with (x -> x * 2)",
        {}
    )

    assert len(result) == len(numbers)
```

---

## 7. Migration Path

### From Raw exec() / eval()

If you're currently using Python's `exec()` for LLM-generated code:

**Before (dangerous):**
```python
llm_code = get_code_from_llm(task)
exec(llm_code, {"data": my_data})  # 😱
```

**After (safe):**
```python
hoist_code = get_hoist_from_llm(task)  # Use Hoist system prompt
result = runtime.eval(hoist_code, {"data": my_data})  # ✅
```

### Migration Checklist

1. **Audit current LLM code generation** - What operations does generated code actually use?
2. **Map to Hoist equivalents** - Most data transformations map directly
3. **Identify gaps** - What can't be expressed in Hoist? (Usually file I/O, network calls)
4. **Refactor architecture** - Move I/O outside the sandbox, pass data in as context
5. **Update prompts** - Replace code generation prompts with Hoist prompts
6. **Add tests** - Verify behavior matches previous implementation
7. **Deploy with monitoring** - Watch for runtime errors, iterate on prompts

### Common Migration Patterns

**File reading → Context variable:**
```python
# Before: LLM generates code that reads files
# exec("content = open('data.txt').read()")

# After: Read file yourself, pass as context
content = open('data.txt').read()
result = runtime.eval(hoist_program, {"content": content})
```

**HTTP requests → Pre-fetch:**
```python
# Before: LLM generates code that fetches URLs
# exec("data = requests.get(url).json()")

# After: Fetch yourself, pass as context
data = requests.get(url).json()
result = runtime.eval(hoist_program, {"data": data})
```

**Database queries → Provide query results:**
```python
# Before: LLM generates SQL (dangerous!)
# exec(f"cursor.execute('{llm_sql}')")

# After: Provide data, let Hoist transform it
rows = safe_query(predefined_query)
result = runtime.eval(hoist_program, {"rows": rows})
```

---

## 8. Security Considerations for Adopters

### Threat Model

Hoist protects against:

| Threat | Mitigation |
|--------|------------|
| Arbitrary code execution | No `eval`, no FFI, no imports |
| File system access | No file operations in language |
| Network access | No HTTP/socket operations |
| Environment variable leakage | No `env` access |
| Denial of service (infinite loops) | Iteration limits, total language |
| Denial of service (memory) | Bounded data structures |
| Regex DoS (ReDoS) | RE2 engine with linear time guarantee |
| Data exfiltration | Only return value leaves sandbox |

### What Hoist Does NOT Protect Against

| Threat | Your Responsibility |
|--------|---------------------|
| Prompt injection | Validate/sanitize inputs to `ask` prompts |
| LLM hallucination | Verify outputs, don't trust blindly |
| Sensitive data in context | Don't pass secrets as context variables |
| Cost attacks (many LLM calls) | Set reasonable limits on `ask` calls |
| LLM response manipulation | The LLM callback is your code—secure it |

### Secure Configuration

```python
runtime = hoist.Runtime(
    ask=my_llm,

    # Execution limits
    max_iterations=10_000,      # Prevent long-running programs
    max_recursion=100,          # Prevent deep call stacks
    timeout_ms=30_000,          # Hard timeout

    # Memory limits
    max_string_length=1_000_000,    # 1MB strings max
    max_list_length=100_000,        # 100k items max
    max_total_memory=50_000_000,    # 50MB total

    # LLM limits
    max_ask_calls=50,           # Limit LLM API calls
    max_ask_prompt_length=10_000,  # Limit prompt size
)
```

### Input Validation

Always validate context before passing to Hoist:

```python
def safe_eval(program: str, context: dict) -> Any:
    # Validate program size
    if len(program) > 10_000:
        raise ValueError("Program too large")

    # Validate context
    for key, value in context.items():
        if isinstance(value, str) and len(value) > 1_000_000:
            raise ValueError(f"Context variable '{key}' too large")

    # Validate no sensitive keys leaked
    sensitive_patterns = ['password', 'secret', 'token', 'key']
    for key in context.keys():
        if any(p in key.lower() for p in sensitive_patterns):
            raise ValueError(f"Potentially sensitive context key: {key}")

    return runtime.eval(program, context)
```

---

## 9. Performance Tuning

### Benchmarks

Typical performance characteristics:

| Operation | Time |
|-----------|------|
| Parse 100-line program | < 1ms |
| Type check | < 1ms |
| `map` over 10k items | ~10ms |
| `filter` over 10k items | ~5ms |
| String operations | ~1μs per call |
| `ask` call | Depends on LLM (typically 500ms-5s) |

**The bottleneck is almost always LLM latency, not Hoist execution.**

### Optimizing LLM Calls

```python
# Bad: Many small calls
"""
let words = split text by " "
return map words with (w -> ask("Define: {w}"))  // N calls!
"""

# Good: Batch into fewer calls
"""
let chunks = chunk(words, 10)  // Group words
return flatten(map chunks with (c ->
    parse_json(ask("Define these words as JSON array: {join(c, ', ')}"))
))
"""
```

### Caching

Implement caching in your `ask` callback:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_llm(prompt: str) -> str:
    return call_llm(prompt)

runtime = hoist.Runtime(ask=cached_llm)
```

Or use a persistent cache:

```python
import redis

cache = redis.Redis()

def cached_ask(prompt: str) -> str:
    key = f"hoist:ask:{hashlib.sha256(prompt.encode()).hexdigest()}"

    cached = cache.get(key)
    if cached:
        return cached.decode()

    response = call_llm(prompt)
    cache.setex(key, 3600, response)  # Cache for 1 hour
    return response
```

### Parallel LLM Calls

For programs with independent `ask` calls, the runtime can parallelize:

```python
runtime = hoist.Runtime(
    ask=my_async_llm,
    parallel_ask=True,      # Enable parallel execution
    max_parallel_asks=5     # Limit concurrent calls
)
```

This transforms:
```hoist
map items with (x -> ask("Process: {x}"))
```

Into parallel calls rather than sequential, dramatically reducing latency for large lists.

---

## 10. Common Pitfalls

### Pitfall 1: Forgetting the Return Statement

```hoist
# Wrong - no return, result is null
let x = 1 + 2

# Right
let x = 1 + 2
return x
```

### Pitfall 2: String Escaping in Prompts

```python
# Wrong - Python escapes vs Hoist escapes
program = "return split context by '\n'"  # Python interprets \n

# Right - raw string or double escape
program = r"return split context by '\n'"
program = "return split context by '\\n'"
```

### Pitfall 3: Expecting Mutation

```hoist
# Wrong - this doesn't modify items
let items = [1, 2, 3]
map items with (x -> x * 2)  # Result discarded!
return items  # Still [1, 2, 3]

# Right - capture the result
let items = [1, 2, 3]
let doubled = map items with (x -> x * 2)
return doubled  # [2, 4, 6]
```

### Pitfall 4: Type Mismatches with JSON

```hoist
# OLD WAY - The LLM might return JSON as a string (fragile)
let response = ask "Return a JSON array of names"
# response = "[\"Alice\", \"Bob\"]" (string, not list!)
let names = parse_json(response)  # May fail!
return map names with upper

# NEW WAY - Use typed output (robust)
let names = ask "Return a JSON array of names" as List<String>
return map names with upper
# Runtime handles formatting, parsing, and retries automatically
```

### Pitfall 5: Assuming ask() is Deterministic

```hoist
# The same prompt can return different results!
let a = ask("Generate a random name")
let b = ask("Generate a random name")
# a and b might differ
```

For reproducibility, make prompts specific or cache results.

### Pitfall 6: Not Handling Empty Lists

```hoist
# This crashes if chunks is empty
let chunks = split context by "\n\n"
return first(chunks)  # Error: empty list

# Right - handle empty case
let chunks = split context by "\n\n"
return if len(chunks) > 0 then first(chunks) else ""
```

### Pitfall 7: Not Using AI-Friendly Features

```hoist
# Fragile - manual error handling
let response = ask "Extract data"
let data = if contains(response, "error") then
    ask "Try again: extract data"  # Manual retry
else
    response
let parsed = parse_json(data)  # May still fail!

# Robust - use built-in features
let data = ask "Extract data" as {name: String, value: Int}
    with retries: 3
    fallback {name: "unknown", value: 0}
```

### Pitfall 8: Not Leveraging Channels

```hoist
# Expensive - using smart model for everything
let summaries = map chunks with ask "Brief summary: {it}"  # Overkill!
let analysis = ask "Deep analysis: {join(summaries, ' ')}"

# Cost-effective - route appropriately
let summaries = map chunks with ask "Brief summary: {it}" via fast
let analysis = ask "Deep analysis: {join(summaries, ' ')}" via smart
```

---

## 11. Example Integrations

### FastAPI Endpoint

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import hoist

app = FastAPI()
runtime = hoist.Runtime(ask=call_openai)

class EvalRequest(BaseModel):
    program: str
    context: dict = {}

@app.post("/eval")
async def eval_hoist(request: EvalRequest):
    try:
        result = runtime.eval(request.program, request.context)
        return {"result": result}
    except hoist.ParseError as e:
        raise HTTPException(400, detail={"error": "parse", "message": str(e)})
    except hoist.RuntimeError as e:
        raise HTTPException(400, detail={"error": "runtime", "message": str(e)})
```

### LangChain Integration

```python
from langchain.tools import Tool
import hoist

runtime = hoist.Runtime(ask=llm.invoke)

hoist_tool = Tool(
    name="hoist_eval",
    description="Execute a Hoist program to process data",
    func=lambda input: runtime.eval(
        input["program"],
        input["context"]
    )
)

# Use in an agent
agent = create_agent(llm, [hoist_tool])
```

### Celery Task

```python
from celery import Celery
import hoist

app = Celery('tasks')

@app.task(bind=True, max_retries=3)
def run_hoist(self, program: str, context: dict):
    runtime = hoist.Runtime(
        ask=call_llm,
        timeout_ms=60_000
    )

    try:
        return runtime.eval(program, context)
    except hoist.HostError as e:
        if e.retryable:
            raise self.retry(exc=e, countdown=5)
        raise
```

### Jupyter Notebook Magic

```python
# In your package: hoist/jupyter.py
from IPython.core.magic import register_cell_magic
import hoist

runtime = hoist.Runtime(ask=lambda p: input(f"LLM prompt: {p}\nResponse: "))

@register_cell_magic
def hoist_eval(line, cell):
    """Run Hoist code in a notebook cell."""
    context = get_ipython().user_ns.get('hoist_context', {})
    result = runtime.eval(cell, context)
    return result

# Usage in notebook:
# %load_ext hoist.jupyter
#
# %%hoist
# let nums = [1, 2, 3, 4, 5]
# return map nums with (x -> x * 2)
```

---

## 12. Versioning & Compatibility

### Semantic Versioning

Hoist follows semver:

- **Major (1.0 → 2.0):** Breaking changes to language syntax or semantics
- **Minor (1.0 → 1.1):** New stdlib functions, new features (backwards compatible)
- **Patch (1.0.0 → 1.0.1):** Bug fixes, performance improvements

### Language Version Header

Programs can optionally declare required version:

```hoist
#!hoist 1.0

let x = 42
return x
```

The runtime will error if the version is incompatible.

### Deprecation Policy

- Deprecated features emit warnings for 2 minor versions
- Removed in the next major version
- Migration guides provided for all breaking changes

### Compatibility Matrix

| Hoist Version | Python | Node.js | Browsers |
|---------------|--------|---------|----------|
| 1.0.x | 3.9+ | 18+ | ES2020+ |
| 1.1.x | 3.9+ | 18+ | ES2020+ |

---

## 13. Support & Community

### Getting Help

- **Documentation:** https://hoist-lang.dev/docs
- **GitHub Issues:** Bug reports and feature requests
- **Discord:** Real-time community support
- **Stack Overflow:** Tag questions with `hoist-lang`

### Contributing

- **Report bugs:** Include minimal reproduction
- **Suggest features:** Open a GitHub discussion first
- **Submit PRs:** See CONTRIBUTING.md
- **Add stdlib functions:** Propose in discussions, implement if approved

### Roadmap Visibility

Public roadmap at https://github.com/anthropic/hoist/projects/1

Planned features (subject to change):
- [ ] Language server protocol (LSP) for IDE support
- [ ] Playground website
- [ ] More stdlib functions (date/time, advanced string ops)
- [ ] Debugger integration
- [ ] Visual program builder

---

## Appendix: Package Contents

### Python Package Structure

```
hoist/
├── __init__.py              # Main API: Runtime, AsyncRuntime
├── _native.cpython-311-*.so # Compiled Rust binary
├── errors.py                # Exception classes
├── prompts/
│   ├── __init__.py          # Prompt access API
│   ├── system_prompt.txt
│   ├── reference_card.txt
│   └── few_shot_examples.txt
├── py.typed                 # PEP 561 marker
└── types.pyi                # Type stubs
```

### npm Package Structure

```
@anthropic/hoist/
├── package.json
├── index.js                 # Main entry point
├── index.d.ts               # TypeScript definitions
├── native.node              # Compiled binary (platform-specific)
└── prompts/
    ├── system-prompt.txt
    ├── reference-card.txt
    └── few-shot-examples.txt
```

### What Ships Where

| Asset | Python | npm | WASM |
|-------|--------|-----|------|
| Compiled runtime | ✅ `.so`/`.pyd` | ✅ `.node` | ✅ `.wasm` |
| Type definitions | ✅ `.pyi` | ✅ `.d.ts` | ✅ `.d.ts` |
| Prompt templates | ✅ | ✅ | ✅ |
| Source maps | ❌ | ❌ | ✅ |
| Debug symbols | ❌ (separate pkg) | ❌ | ❌ |
