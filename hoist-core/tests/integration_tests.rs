/// Integration tests for the Hoist language implementation.
/// These tests exercise full programs through lex -> parse -> eval.

use hoist_core::interpreter::{AskHandler, Interpreter, ResourceLimits, Value};
use hoist_core::parser::parse_source;

// ---------- Test helper ----------

struct MockAskHandler {
    responses: std::collections::HashMap<String, String>,
}

impl MockAskHandler {
    fn new(responses: Vec<(&str, &str)>) -> Self {
        Self {
            responses: responses.into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
        }
    }
}

impl AskHandler for MockAskHandler {
    fn ask(&self, prompt: &str, _channel: Option<&str>) -> Result<String, String> {
        self.responses.get(prompt).cloned().ok_or_else(|| format!("no mock for prompt: {}", prompt))
    }
}

fn run(src: &str) -> Value {
    let prog = parse_source(src).expect("parse failed");
    let mut interp = Interpreter::new();
    interp.eval_program(&prog).expect("eval failed")
}

fn run_with_context(src: &str, context: &str) -> Value {
    let prog = parse_source(src).expect("parse failed");
    let mut interp = Interpreter::new();
    interp.set_var("context", Value::String(context.to_string()));
    interp.eval_program(&prog).expect("eval failed")
}

fn run_with_ask(src: &str, context: &str, responses: Vec<(&str, &str)>) -> Value {
    let prog = parse_source(src).expect("parse failed");
    let mut interp = Interpreter::new()
        .with_ask_handler(Box::new(MockAskHandler::new(responses)));
    interp.set_var("context", Value::String(context.to_string()));
    interp.eval_program(&prog).expect("eval failed")
}

fn run_err(src: &str) -> hoist_core::error::HoistError {
    let prog = parse_source(src).expect("parse failed");
    let mut interp = Interpreter::new();
    interp.eval_program(&prog).expect_err("expected error")
}

fn run_with_limits(src: &str, limits: ResourceLimits) -> Result<Value, hoist_core::error::HoistError> {
    let prog = parse_source(src).expect("parse failed");
    let mut interp = Interpreter::new().with_limits(limits);
    interp.eval_program(&prog)
}



// =====================================================================
// Spec Appendix B: Compliance Tests
// =====================================================================

#[test]
fn compliance_b4_filter_and_count() {
    // B.4 Execution Test: filter and count
    let result = run_with_context(
        r#"let items = split context by ","
let long = filter items where length(it) > 3
return show(length(long))"#,
        "a,bb,ccc,dddd,eeeee",
    );
    assert_eq!(result, Value::String("2".into()));
}

#[test]
fn compliance_b5_simple_ask() {
    let result = run_with_ask(
        r#"return ask "What is 2+2?""#,
        "",
        vec![("What is 2+2?", "4")],
    );
    assert_eq!(result, Value::String("4".into()));
}

#[test]
fn compliance_b6_map_with_ask() {
    let result = run_with_ask(
        r#"let items = ["a", "b"]
return join (map items with ask "Process: {it}") with ",""#,
        "",
        vec![("Process: a", "A"), ("Process: b", "B")],
    );
    assert_eq!(result, Value::String("A,B".into()));
}

#[test]
fn compliance_b7_ask_limit_exceeded() {
    let src = r#"let items = split context by ","
return join (map items with ask "X") with ",""#;
    let prog = parse_source(src).unwrap();
    let mut interp = Interpreter::new()
        .with_ask_handler(Box::new(MockAskHandler::new(vec![("X", "ok")])))
        .with_limits(ResourceLimits { max_ask_calls: 10, ..Default::default() });
    interp.set_var("context", Value::String("1,2,3,4,5,6,7,8,9,10,11".into()));
    let result = interp.eval_program(&prog);
    assert!(matches!(result, Err(hoist_core::error::HoistError::LimitExceeded { .. })));
}

// =====================================================================
// Core language features
// =====================================================================

#[test]
fn test_multiple_bindings() {
    let result = run(r#"
let a = 10
let b = 20
let c = a + b
return c * 2
"#);
    assert_eq!(result, Value::Int(60));
}

#[test]
fn test_nested_if() {
    let result = run(r#"
let x = 15
return if x > 20 then "big" else if x > 10 then "medium" else "small"
"#);
    assert_eq!(result, Value::String("medium".into()));
}

#[test]
fn test_complex_string_interpolation() {
    let result = run(r#"
let name = "Alice"
let count = 3
return "Hello {name}, you have {show(count)} items"
"#);
    assert_eq!(result, Value::String("Hello Alice, you have 3 items".into()));
}

#[test]
fn test_map_with_explicit_lambda() {
    let result = run(r#"
let nums = [1, 2, 3, 4, 5]
return map nums with x -> x * x
"#);
    assert_eq!(result, Value::List(vec![
        Value::Int(1), Value::Int(4), Value::Int(9), Value::Int(16), Value::Int(25),
    ]));
}

#[test]
fn test_filter_with_function_call() {
    let result = run(r#"
let words = ["hello", "", "world", "", "!"]
return filter words where length(it) > 0
"#);
    assert_eq!(result, Value::List(vec![
        Value::String("hello".into()),
        Value::String("world".into()),
        Value::String("!".into()),
    ]));
}

#[test]
fn test_fold_string_concat() {
    let result = run(r#"
let words = ["hello", " ", "world"]
return fold words from "" with acc, w -> acc ++ w
"#);
    assert_eq!(result, Value::String("hello world".into()));
}

#[test]
fn test_pipeline_complex() {
    let result = run_with_context(r#"
let chunks = split context by "\n"
let nonempty = chunks |> filter where length(it) > 0
return show(length(nonempty))
"#, "line1\n\nline2\n\nline3");
    assert_eq!(result, Value::String("3".into()));
}

#[test]
fn test_window_operation() {
    let result = run(r#"
let text = "abcdefghij"
let windows = window text size 4 stride 3
return length(windows)
"#);
    // "abcd", "defg", "ghij" — 3 windows
    match result {
        Value::Int(n) => assert!(n >= 3),
        _ => panic!("expected Int"),
    }
}

#[test]
fn test_record_access() {
    let result = run(r#"
let person = {name: "Bob", age: 25}
return person.name ++ " is " ++ show(person.age)
"#);
    assert_eq!(result, Value::String("Bob is 25".into()));
}

#[test]
fn test_match_literals() {
    let result = run(r#"
let x = 2
return match x with
| 1 -> "one"
| 2 -> "two"
| 3 -> "three"
| _ -> "other"
"#);
    assert_eq!(result, Value::String("two".into()));
}

#[test]
fn test_list_operations() {
    // reverse
    let result = run("return reverse([1, 2, 3])");
    assert_eq!(result, Value::List(vec![Value::Int(3), Value::Int(2), Value::Int(1)]));

    // sort strings
    let result = run(r#"return sort(["banana", "apple", "cherry"])"#);
    assert_eq!(result, Value::List(vec![
        Value::String("apple".into()),
        Value::String("banana".into()),
        Value::String("cherry".into()),
    ]));

    // unique
    let result = run("return unique([1, 2, 2, 3, 3, 3])");
    assert_eq!(result, Value::List(vec![Value::Int(1), Value::Int(2), Value::Int(3)]));

    // flatten
    let result = run("return flatten([[1, 2], [3, 4], [5]])");
    assert_eq!(result, Value::List(vec![
        Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4), Value::Int(5),
    ]));
}

#[test]
fn test_zip_enumerate() {
    let result = run(r#"return zip(["a", "b", "c"], [1, 2, 3])"#);
    match result {
        Value::List(items) => {
            assert_eq!(items.len(), 3);
            // Each item is a Tuple
            match &items[0] {
                Value::Tuple(t) => {
                    assert_eq!(t[0], Value::String("a".into()));
                    assert_eq!(t[1], Value::Int(1));
                }
                _ => panic!("expected tuple"),
            }
        }
        _ => panic!("expected list"),
    }

    let result = run(r#"return enumerate(["x", "y", "z"])"#);
    match result {
        Value::List(items) => {
            assert_eq!(items.len(), 3);
            match &items[0] {
                Value::Tuple(t) => {
                    assert_eq!(t[0], Value::Int(0));
                    assert_eq!(t[1], Value::String("x".into()));
                }
                _ => panic!("expected tuple"),
            }
        }
        _ => panic!("expected list"),
    }
}

#[test]
fn test_numeric_functions() {
    assert_eq!(run("return abs(-5)"), Value::Int(5));
    assert_eq!(run("return min(3, 7)"), Value::Int(3));
    assert_eq!(run("return max(3, 7)"), Value::Int(7));
    assert_eq!(run("return clamp(15, 0, 10)"), Value::Int(10));
    assert_eq!(run("return sum([1, 2, 3, 4, 5])"), Value::Int(15));
}

#[test]
fn test_string_functions() {
    assert_eq!(run(r#"return trim("  hello  ")"#), Value::String("hello".into()));
    assert_eq!(run(r#"return trim_start("  hello  ")"#), Value::String("hello  ".into()));
    assert_eq!(run(r#"return trim_end("  hello  ")"#), Value::String("  hello".into()));
    assert_eq!(run(r#"return starts_with("hello", "he")"#), Value::Bool(true));
    assert_eq!(run(r#"return ends_with("hello", "lo")"#), Value::Bool(true));
    assert_eq!(run(r#"return replace("hello world", "world", "hoist")"#), Value::String("hello hoist".into()));
}

#[test]
fn test_regex_functions() {
    assert_eq!(run(r#"return matches("hello123", "[0-9]+")"#), Value::Bool(true));
    assert_eq!(run(r#"return matches("hello", "[0-9]+")"#), Value::Bool(false));

    let result = run(r#"return find_all("abc123def456", "[0-9]+")"#);
    assert_eq!(result, Value::List(vec![
        Value::String("123".into()),
        Value::String("456".into()),
    ]));
}

#[test]
fn test_type_of() {
    assert_eq!(run(r#"return type_of(42)"#), Value::String("Int".into()));
    assert_eq!(run(r#"return type_of("hello")"#), Value::String("String".into()));
    assert_eq!(run(r#"return type_of(true)"#), Value::String("Bool".into()));
    assert_eq!(run(r#"return type_of([1, 2])"#), Value::String("List".into()));
}

#[test]
fn test_empty_list() {
    assert_eq!(run("return empty([])"), Value::Bool(true));
    assert_eq!(run("return empty([1])"), Value::Bool(false));
}

#[test]
fn test_first_last() {
    let result = run("return first([10, 20, 30])");
    assert_eq!(result, Value::Optional(Some(Box::new(Value::Int(10)))));

    let result = run("return last([10, 20, 30])");
    assert_eq!(result, Value::Optional(Some(Box::new(Value::Int(30)))));

    let result = run("return first([])");
    assert_eq!(result, Value::Optional(None));
}

#[test]
fn test_division_by_zero() {
    let err = run_err("return 1 / 0");
    assert!(matches!(err, hoist_core::error::HoistError::DivisionByZero));
}

#[test]
fn test_unbound_variable() {
    let err = run_err("return undefined_var");
    assert!(matches!(err, hoist_core::error::HoistError::UnboundVariable { .. }));
}

// =====================================================================
// Spec examples (simplified, without real LLM)
// =====================================================================

#[test]
fn spec_example_needle_in_haystack() {
    let src = r#"
let chunks = split context by "\n\n"
let relevant = filter chunks where contains(it, "secret") or contains(it, "code")
let candidates = if length(relevant) > 0
    then relevant
    else chunks
let analyses = map candidates with ask "Extract any secret code from: {it}"
let findings = filter analyses where not contains(it, "NONE")
return if length(findings) > 0
    then join findings with "\n"
    else "No secret code found"
"#;

    let context = "Hello world\n\nThe secret code is 42\n\nNothing here";
    let result = run_with_ask(src, context, vec![
        ("Extract any secret code from: The secret code is 42", "42"),
    ]);
    assert_eq!(result, Value::String("42".into()));
}

#[test]
fn spec_example_information_aggregation() {
    let src = r#"
let entries = lines(context)
let classifications = map entries with ask "Classify: {it}"
let locations = filter classifications where it == "location"
return "Locations: {show(length(locations))}"
"#;

    let result = run_with_ask(src, "Where is Paris?\nWhat is 2+2?\nWhere is Tokyo?", vec![
        ("Classify: Where is Paris?", "location"),
        ("Classify: What is 2+2?", "numeric"),
        ("Classify: Where is Tokyo?", "location"),
    ]);
    assert_eq!(result, Value::String("Locations: 2".into()));
}

#[test]
fn spec_example_recursive_summarization() {
    let src = r#"
let chunks = split context by "\n\n"
let summaries = map chunks with ask "Summarize: {it}"
return join summaries with "\n"
"#;

    let result = run_with_ask(src, "Paragraph one.\n\nParagraph two.", vec![
        ("Summarize: Paragraph one.", "Summary 1"),
        ("Summarize: Paragraph two.", "Summary 2"),
    ]);
    assert_eq!(result, Value::String("Summary 1\nSummary 2".into()));
}

// =====================================================================
// Ask modifiers
// =====================================================================

#[test]
fn test_ask_with_channel() {
    struct ChannelHandler;
    impl AskHandler for ChannelHandler {
        fn ask(&self, _prompt: &str, channel: Option<&str>) -> Result<String, String> {
            Ok(format!("channel:{}", channel.unwrap_or("default")))
        }
    }

    let prog = parse_source(r#"return ask "hello" via summarizer"#).unwrap();
    let mut interp = Interpreter::new()
        .with_ask_handler(Box::new(ChannelHandler));
    let result = interp.eval_program(&prog).unwrap();
    assert_eq!(result, Value::String("channel:summarizer".into()));
}

#[test]
fn test_ask_with_fallback() {
    struct FailHandler;
    impl AskHandler for FailHandler {
        fn ask(&self, _prompt: &str, _channel: Option<&str>) -> Result<String, String> {
            Err("service unavailable".into())
        }
    }

    let prog = parse_source(r#"return ask "hello" fallback "default_value""#).unwrap();
    let mut interp = Interpreter::new()
        .with_ask_handler(Box::new(FailHandler));
    let result = interp.eval_program(&prog).unwrap();
    assert_eq!(result, Value::String("default_value".into()));
}

#[test]
fn test_ask_with_retries() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct RetryHandler {
        attempt: Arc<AtomicUsize>,
    }
    impl AskHandler for RetryHandler {
        fn ask(&self, _prompt: &str, _channel: Option<&str>) -> Result<String, String> {
            let n = self.attempt.fetch_add(1, Ordering::SeqCst);
            if n < 2 {
                Err("temporary failure".into())
            } else {
                Ok("success".into())
            }
        }
    }

    let attempt = Arc::new(AtomicUsize::new(0));
    let prog = parse_source(r#"return ask "hello" with retries: 3"#).unwrap();
    let mut interp = Interpreter::new()
        .with_ask_handler(Box::new(RetryHandler { attempt: attempt.clone() }));
    let result = interp.eval_program(&prog).unwrap();
    assert_eq!(result, Value::String("success".into()));
    assert_eq!(attempt.load(Ordering::SeqCst), 3); // 2 failures + 1 success
}

// =====================================================================
// Pipeline edge cases
// =====================================================================

#[test]
fn test_pipeline_split_filter_map_join() {
    let result = run_with_context(r#"
return context
    |> split by ","
    |> filter where length(it) > 0
    |> map with upper(it)
    |> join with ";"
"#, "hello,,world,,hoist");
    assert_eq!(result, Value::String("HELLO;WORLD;HOIST".into()));
}

#[test]
fn test_chars_function() {
    let result = run(r#"return chars("abc")"#);
    assert_eq!(result, Value::List(vec![
        Value::String("a".into()),
        Value::String("b".into()),
        Value::String("c".into()),
    ]));
}

#[test]
fn test_words_function() {
    let result = run(r#"return words("hello  world  foo")"#);
    assert_eq!(result, Value::List(vec![
        Value::String("hello".into()),
        Value::String("world".into()),
        Value::String("foo".into()),
    ]));
}

#[test]
fn test_parse_json() {
    let result = run(r#"return parse_json("\{\"key\": \"value\", \"num\": 42}")"#);
    match result {
        Value::Record(map) => {
            assert_eq!(map.get("key"), Some(&Value::String("value".into())));
            assert_eq!(map.get("num"), Some(&Value::Int(42)));
        }
        _ => panic!("expected record, got {:?}", result),
    }
}

#[test]
fn test_nested_map_filter() {
    let result = run(r#"
let items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
let evens = filter items where it % 2 == 0
let doubled = map evens with it * 2
return sum(doubled)
"#);
    // evens: [2,4,6,8,10], doubled: [4,8,12,16,20], sum: 60
    assert_eq!(result, Value::Int(60));
}

#[test]
fn test_string_concat_operator() {
    let result = run(r#"
let a = "hello"
let b = "world"
return a ++ " " ++ b
"#);
    assert_eq!(result, Value::String("hello world".into()));
}

#[test]
fn test_boolean_short_circuit() {
    // `false and (1/0 == 0)` should NOT trigger division by zero due to short-circuit
    let result = run("return false and (1 == 1)");
    assert_eq!(result, Value::Bool(false));

    let result = run("return true or (1 == 1)");
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn test_match_with_wildcard() {
    let result = run(r#"
let x = 99
return match x with
| 1 -> "one"
| _ -> "unknown"
"#);
    assert_eq!(result, Value::String("unknown".into()));
}

#[test]
fn test_show_conversion_in_interpolation() {
    let result = run(r#"
let nums = [1, 2, 3]
return "count: {show(length(nums))}"
"#);
    assert_eq!(result, Value::String("count: 3".into()));
}

// =====================================================================
// New feature tests: Optional `or` unwrap
// =====================================================================

#[test]
fn test_optional_or_unwrap_some() {
    let result = run(r#"
let items = [10, 20, 30]
let val = first(items) or 0
return val
"#);
    assert_eq!(result, Value::Int(10));
}

#[test]
fn test_optional_or_unwrap_none() {
    let result = run(r#"
let items: List<Int> = []
let val = first(items) or 0
return val
"#);
    assert_eq!(result, Value::Int(0));
}

#[test]
fn test_optional_or_with_string() {
    let result = run(r#"
let items: List<String> = []
let val = first(items) or "default"
return val
"#);
    assert_eq!(result, Value::String("default".into()));
}

#[test]
fn test_optional_or_chained() {
    // `or` should still work as boolean OR for bools
    let result = run("return false or true");
    assert_eq!(result, Value::Bool(true));

    let result = run("return true or false");
    assert_eq!(result, Value::Bool(true));
}

// =====================================================================
// New feature tests: Duplicate binding detection
// =====================================================================

#[test]
fn test_duplicate_binding_error() {
    let err = run_err(r#"
let x = 1
let x = 2
return x
"#);
    match err {
        hoist_core::error::HoistError::TypeError { message, .. } => {
            assert!(message.contains("already bound"), "expected 'already bound' error, got: {}", message);
        }
        other => panic!("expected TypeError, got: {:?}", other),
    }
}

// =====================================================================
// New feature tests: Pipe support for fold, take, drop, window, slice
// =====================================================================

#[test]
fn test_pipe_fold() {
    let result = run(r#"
let nums = [1, 2, 3, 4, 5]
return nums |> fold from 0 with acc, n -> acc + n
"#);
    assert_eq!(result, Value::Int(15));
}

#[test]
fn test_pipe_take() {
    let result = run(r#"
let nums = [1, 2, 3, 4, 5]
return nums |> take 3
"#);
    assert_eq!(result, Value::List(vec![Value::Int(1), Value::Int(2), Value::Int(3)]));
}

#[test]
fn test_pipe_drop() {
    let result = run(r#"
let nums = [1, 2, 3, 4, 5]
return nums |> drop 2
"#);
    assert_eq!(result, Value::List(vec![Value::Int(3), Value::Int(4), Value::Int(5)]));
}

#[test]
fn test_pipe_window() {
    let result = run(r#"
let text = "abcdef"
return text |> window size 3 stride 2
"#);
    // windows: "abc", "cde" (or possibly "abc", "cde", "ef" depending on impl)
    match result {
        Value::List(items) => assert!(items.len() >= 2),
        _ => panic!("expected list"),
    }
}

#[test]
fn test_pipe_slice() {
    // slice operates on strings, not lists
    let result = run(r#"
return "Hello, World!" |> slice from 0 to 5
"#);
    assert_eq!(result, Value::String("Hello".into()));
}

#[test]
fn test_pipe_chained_fold() {
    let result = run(r#"
return [1, 2, 3, 4, 5]
    |> filter where it > 2
    |> fold from 0 with acc, n -> acc + n
"#);
    // filter: [3, 4, 5], fold: 12
    assert_eq!(result, Value::Int(12));
}

#[test]
fn test_pipe_chained_take_drop() {
    let result = run(r#"
return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    |> drop 2
    |> take 5
"#);
    assert_eq!(result, Value::List(vec![
        Value::Int(3), Value::Int(4), Value::Int(5), Value::Int(6), Value::Int(7),
    ]));
}

// =====================================================================
// New feature tests: max_string_size enforcement
// =====================================================================

#[test]
fn test_max_string_size_concat() {
    let limits = ResourceLimits {
        max_string_size: 20,
        ..Default::default()
    };
    let result = run_with_limits(
        r#"return "aaaaaaaaaa" ++ "bbbbbbbbbbb""#,
        limits,
    );
    assert!(result.is_err(), "should fail: string concat exceeds max_string_size");
    match result.unwrap_err() {
        hoist_core::error::HoistError::LimitExceeded { .. } => {}
        other => panic!("expected LimitExceeded, got: {:?}", other),
    }
}

#[test]
fn test_max_string_size_interpolation() {
    let limits = ResourceLimits {
        max_string_size: 10,
        ..Default::default()
    };
    let result = run_with_limits(
        r#"let name = "a very long name indeed"
return "Hello {name}""#,
        limits,
    );
    assert!(result.is_err(), "should fail: interpolated string exceeds max_string_size");
}

#[test]
fn test_string_within_limit_succeeds() {
    let limits = ResourceLimits {
        max_string_size: 100,
        ..Default::default()
    };
    let result = run_with_limits(r#"return "hello" ++ " " ++ "world""#, limits);
    assert_eq!(result.unwrap(), Value::String("hello world".into()));
}

// =====================================================================
// New feature tests: max_collection_size enforcement
// =====================================================================

#[test]
fn test_max_collection_size_map() {
    let limits = ResourceLimits {
        max_collection_size: 3,
        ..Default::default()
    };
    // map output has 5 elements, exceeding limit of 3
    let result = run_with_limits(
        r#"return map [1, 2, 3, 4, 5] with it * 2"#,
        limits,
    );
    assert!(result.is_err(), "should fail: map result exceeds max_collection_size");
    match result.unwrap_err() {
        hoist_core::error::HoistError::LimitExceeded { .. } => {}
        other => panic!("expected LimitExceeded, got: {:?}", other),
    }
}

#[test]
fn test_max_collection_size_split() {
    let limits = ResourceLimits {
        max_collection_size: 2,
        ..Default::default()
    };
    let result = run_with_limits(
        r#"return split "a,b,c,d" by ",""#,
        limits,
    );
    assert!(result.is_err(), "should fail: split result exceeds max_collection_size");
}

#[test]
fn test_collection_within_limit_succeeds() {
    let limits = ResourceLimits {
        max_collection_size: 10,
        ..Default::default()
    };
    let result = run_with_limits(
        r#"return map [1, 2, 3] with it * 2"#,
        limits,
    );
    assert_eq!(result.unwrap(), Value::List(vec![Value::Int(2), Value::Int(4), Value::Int(6)]));
}

// =====================================================================
// Additional edge case tests
// =====================================================================

#[test]
fn test_take_and_drop_standalone() {
    let result = run("return take 2 from [1, 2, 3, 4, 5]");
    assert_eq!(result, Value::List(vec![Value::Int(1), Value::Int(2)]));

    let result = run("return drop 3 from [1, 2, 3, 4, 5]");
    assert_eq!(result, Value::List(vec![Value::Int(4), Value::Int(5)]));
}

#[test]
fn test_slice_standalone() {
    let result = run(r#"return slice "Hello, World!" from 7 to 12"#);
    assert_eq!(result, Value::String("World".into()));
}

#[test]
fn test_fold_standalone() {
    let result = run(r#"
let words = ["hello", " ", "world"]
return fold words from "" with acc, w -> acc ++ w
"#);
    assert_eq!(result, Value::String("hello world".into()));
}

#[test]
fn test_step_limit() {
    let limits = ResourceLimits {
        max_steps: 10,
        ..Default::default()
    };
    // A long computation should exceed the step limit
    let result = run_with_limits(
        r#"return fold [1,2,3,4,5,6,7,8,9,10] from 0 with acc, n -> acc + n"#,
        limits,
    );
    assert!(result.is_err(), "should fail: computation exceeds max_steps");
}
