use std::collections::HashMap;
use crate::error::HoistError;
use crate::interpreter::{Interpreter, Value, Env};

/// Register all builtin functions into the environment.
pub fn register_builtins(env: &mut Env) {
    let builtins = [
        // String functions
        "length", "upper", "lower", "trim", "trim_start", "trim_end",
        "contains", "starts_with", "ends_with", "replace",
        "chars", "lines", "words",
        // Regex
        "matches", "find", "find_all", "capture",
        // List functions
        "empty", "first", "last", "reverse", "sort", "unique",
        "flatten", "zip", "enumerate",
        // Numeric
        "abs", "min", "max", "clamp", "sum",
        // Conversion
        "show", "parse_int", "parse_json",
        // Utility
        "debug", "type_of",
    ];

    for name in builtins {
        env.insert(name.to_string(), Value::BuiltinFn(name.to_string()));
    }

    // Also register Some as a builtin
    env.insert("Some".to_string(), Value::BuiltinFn("Some".to_string()));
}

/// Dispatch a builtin function call.
pub fn call_builtin(
    _interp: &mut Interpreter,
    name: &str,
    args: Vec<Value>,
) -> Result<Value, HoistError> {
    match name {
        // --- String functions ---
        "length" => {
            check_arity(name, &args, 1)?;
            match &args[0] {
                Value::String(s) => Ok(Value::Int(s.chars().count() as i64)),
                Value::List(l) => Ok(Value::Int(l.len() as i64)),
                _ => Err(type_err(name, "String or List", &args[0])),
            }
        }

        "upper" => {
            check_arity(name, &args, 1)?;
            Ok(Value::String(args[0].as_string()?.to_uppercase()))
        }

        "lower" => {
            check_arity(name, &args, 1)?;
            Ok(Value::String(args[0].as_string()?.to_lowercase()))
        }

        "trim" => {
            check_arity(name, &args, 1)?;
            Ok(Value::String(args[0].as_string()?.trim().to_string()))
        }

        "trim_start" => {
            check_arity(name, &args, 1)?;
            Ok(Value::String(args[0].as_string()?.trim_start().to_string()))
        }

        "trim_end" => {
            check_arity(name, &args, 1)?;
            Ok(Value::String(args[0].as_string()?.trim_end().to_string()))
        }

        "contains" => {
            check_arity(name, &args, 2)?;
            let s = args[0].as_string()?;
            let sub = args[1].as_string()?;
            Ok(Value::Bool(s.contains(sub)))
        }

        "starts_with" => {
            check_arity(name, &args, 2)?;
            let s = args[0].as_string()?;
            let prefix = args[1].as_string()?;
            Ok(Value::Bool(s.starts_with(prefix)))
        }

        "ends_with" => {
            check_arity(name, &args, 2)?;
            let s = args[0].as_string()?;
            let suffix = args[1].as_string()?;
            Ok(Value::Bool(s.ends_with(suffix)))
        }

        "replace" => {
            check_arity(name, &args, 3)?;
            let s = args[0].as_string()?.to_string();
            let old = args[1].as_string()?;
            let new = args[2].as_string()?;
            Ok(Value::String(s.replace(old, new)))
        }

        "chars" => {
            check_arity(name, &args, 1)?;
            let s = args[0].as_string()?;
            let chars: Vec<Value> = s.chars()
                .map(|c| Value::String(c.to_string()))
                .collect();
            Ok(Value::List(chars))
        }

        "lines" => {
            check_arity(name, &args, 1)?;
            let s = args[0].as_string()?;
            let lines: Vec<Value> = s.lines()
                .map(|l| Value::String(l.to_string()))
                .collect();
            Ok(Value::List(lines))
        }

        "words" => {
            check_arity(name, &args, 1)?;
            let s = args[0].as_string()?;
            let words: Vec<Value> = s.split_whitespace()
                .map(|w| Value::String(w.to_string()))
                .collect();
            Ok(Value::List(words))
        }

        // --- Regex functions ---
        "matches" => {
            check_arity(name, &args, 2)?;
            let s = args[0].as_string()?;
            let pattern = args[1].as_string()?;
            let re = regex::Regex::new(pattern)
                .map_err(|e| HoistError::RuntimeError { message: format!("invalid regex: {}", e) })?;
            Ok(Value::Bool(re.is_match(s)))
        }

        "find" => {
            check_arity(name, &args, 2)?;
            let s = args[0].as_string()?;
            let pattern = args[1].as_string()?;
            let re = regex::Regex::new(pattern)
                .map_err(|e| HoistError::RuntimeError { message: format!("invalid regex: {}", e) })?;
            match re.find(s) {
                Some(m) => Ok(Value::Optional(Some(Box::new(Value::String(m.as_str().to_string()))))),
                None => Ok(Value::Optional(None)),
            }
        }

        "find_all" => {
            check_arity(name, &args, 2)?;
            let s = args[0].as_string()?;
            let pattern = args[1].as_string()?;
            let re = regex::Regex::new(pattern)
                .map_err(|e| HoistError::RuntimeError { message: format!("invalid regex: {}", e) })?;
            let matches: Vec<Value> = re.find_iter(s)
                .map(|m| Value::String(m.as_str().to_string()))
                .collect();
            Ok(Value::List(matches))
        }

        "capture" => {
            check_arity(name, &args, 2)?;
            let s = args[0].as_string()?;
            let pattern = args[1].as_string()?;
            let re = regex::Regex::new(pattern)
                .map_err(|e| HoistError::RuntimeError { message: format!("invalid regex: {}", e) })?;
            match re.captures(s) {
                Some(caps) => {
                    let groups: Vec<Value> = caps.iter()
                        .map(|m| match m {
                            Some(m) => Value::String(m.as_str().to_string()),
                            None => Value::String(String::new()),
                        })
                        .collect();
                    Ok(Value::Optional(Some(Box::new(Value::List(groups)))))
                }
                None => Ok(Value::Optional(None)),
            }
        }

        // --- List functions ---
        "empty" => {
            check_arity(name, &args, 1)?;
            let list = args[0].as_list()?;
            Ok(Value::Bool(list.is_empty()))
        }

        "first" => {
            check_arity(name, &args, 1)?;
            let seq = args[0].as_sequence()?;
            Ok(seq.first()
                .map(|v| Value::Optional(Some(Box::new(v.clone()))))
                .unwrap_or(Value::Optional(None)))
        }

        "last" => {
            check_arity(name, &args, 1)?;
            let seq = args[0].as_sequence()?;
            Ok(seq.last()
                .map(|v| Value::Optional(Some(Box::new(v.clone()))))
                .unwrap_or(Value::Optional(None)))
        }

        "reverse" => {
            check_arity(name, &args, 1)?;
            let list = args[0].as_list()?.to_vec();
            let mut reversed = list;
            reversed.reverse();
            Ok(Value::List(reversed))
        }

        "sort" => {
            check_arity(name, &args, 1)?;
            let list = args[0].as_list()?.to_vec();
            let mut sorted = list;
            sorted.sort_by(|a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            Ok(Value::List(sorted))
        }

        "unique" => {
            check_arity(name, &args, 1)?;
            let list = args[0].as_list()?.to_vec();
            let mut seen = Vec::new();
            let mut result = Vec::new();
            for item in list {
                let key = item.display();
                if !seen.contains(&key) {
                    seen.push(key);
                    result.push(item);
                }
            }
            Ok(Value::List(result))
        }

        "flatten" => {
            check_arity(name, &args, 1)?;
            let list = args[0].as_list()?.to_vec();
            let mut result = Vec::new();
            for item in list {
                match item {
                    Value::List(inner) => result.extend(inner),
                    other => result.push(other),
                }
            }
            Ok(Value::List(result))
        }

        "zip" => {
            check_arity(name, &args, 2)?;
            let a = args[0].as_list()?.to_vec();
            let b = args[1].as_list()?.to_vec();
            let pairs: Vec<Value> = a.into_iter().zip(b)
                .map(|(x, y)| Value::Tuple(vec![x, y]))
                .collect();
            Ok(Value::List(pairs))
        }

        "enumerate" => {
            check_arity(name, &args, 1)?;
            let list = args[0].as_list()?.to_vec();
            let enumerated: Vec<Value> = list.into_iter().enumerate()
                .map(|(i, v)| Value::Tuple(vec![Value::Int(i as i64), v]))
                .collect();
            Ok(Value::List(enumerated))
        }

        // --- Numeric functions ---
        "abs" => {
            check_arity(name, &args, 1)?;
            Ok(Value::Int(args[0].as_int()?.abs()))
        }

        "min" => {
            check_arity(name, &args, 2)?;
            let a = args[0].as_int()?;
            let b = args[1].as_int()?;
            Ok(Value::Int(a.min(b)))
        }

        "max" => {
            check_arity(name, &args, 2)?;
            let a = args[0].as_int()?;
            let b = args[1].as_int()?;
            Ok(Value::Int(a.max(b)))
        }

        "clamp" => {
            check_arity(name, &args, 3)?;
            let n = args[0].as_int()?;
            let lo = args[1].as_int()?;
            let hi = args[2].as_int()?;
            Ok(Value::Int(n.clamp(lo, hi)))
        }

        "sum" => {
            check_arity(name, &args, 1)?;
            let list = args[0].as_list()?;
            let mut total: i64 = 0;
            for item in list {
                total += item.as_int()?;
            }
            Ok(Value::Int(total))
        }

        // --- Conversion functions ---
        "show" => {
            check_arity(name, &args, 1)?;
            Ok(Value::String(args[0].display()))
        }

        "parse_int" => {
            check_arity(name, &args, 1)?;
            let s = args[0].as_string()?;
            match s.trim().parse::<i64>() {
                Ok(n) => Ok(Value::Optional(Some(Box::new(Value::Int(n))))),
                Err(_) => Ok(Value::Optional(None)),
            }
        }

        "parse_json" => {
            check_arity(name, &args, 1)?;
            let s = args[0].as_string()?;
            let json: serde_json::Value = serde_json::from_str(s)
                .map_err(|e| HoistError::RuntimeError { message: format!("JSON parse error: {}", e) })?;
            Ok(json_to_value(json))
        }

        // --- Utility functions ---
        "debug" => {
            check_arity(name, &args, 1)?;
            eprintln!("[debug] {}", args[0].display());
            Ok(args[0].clone())
        }

        "type_of" => {
            check_arity(name, &args, 1)?;
            Ok(Value::String(args[0].type_name().to_string()))
        }

        // --- Some constructor ---
        "Some" => {
            check_arity(name, &args, 1)?;
            Ok(Value::Optional(Some(Box::new(args[0].clone()))))
        }

        _ => Err(HoistError::RuntimeError {
            message: format!("unknown builtin function '{}'", name),
        }),
    }
}

fn check_arity(_name: &str, args: &[Value], expected: usize) -> Result<(), HoistError> {
    if args.len() != expected {
        Err(HoistError::ArityMismatch {
            expected,
            got: args.len(),
        })
    } else {
        Ok(())
    }
}

fn type_err(func: &str, expected: &str, got: &Value) -> HoistError {
    HoistError::RuntimeTypeError {
        message: format!("{}: expected {}, got {}", func, expected, got.type_name()),
    }
}

pub fn json_to_value(json: serde_json::Value) -> Value {
    match json {
        serde_json::Value::Null => Value::Optional(None),
        serde_json::Value::Bool(b) => Value::Bool(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else {
                Value::String(n.to_string())
            }
        }
        serde_json::Value::String(s) => Value::String(s),
        serde_json::Value::Array(arr) => {
            Value::List(arr.into_iter().map(json_to_value).collect())
        }
        serde_json::Value::Object(obj) => {
            let mut map = HashMap::new();
            for (k, v) in obj {
                map.insert(k, json_to_value(v));
            }
            Value::Record(map)
        }
    }
}
