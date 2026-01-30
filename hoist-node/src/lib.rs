use napi::bindgen_prelude::*;
use napi_derive::napi;
use hoist_core::{parse_source, Interpreter, Value, ResourceLimits};

/// Convert a Hoist Value to a serde_json::Value for napi serialization.
fn value_to_json(val: &Value) -> serde_json::Value {
    match val {
        Value::Int(n) => serde_json::Value::Number(serde_json::Number::from(*n)),
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::String(s) => serde_json::Value::String(s.clone()),
        Value::List(items) => {
            serde_json::Value::Array(items.iter().map(value_to_json).collect())
        }
        Value::Record(map) => {
            let obj: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), value_to_json(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        Value::Tuple(items) => {
            serde_json::Value::Array(items.iter().map(value_to_json).collect())
        }
        Value::Optional(Some(inner)) => value_to_json(inner),
        Value::Optional(None) => serde_json::Value::Null,
        Value::Closure { .. } => serde_json::Value::String("<function>".to_string()),
        Value::BuiltinFn(name) => {
            serde_json::Value::String(format!("<builtin:{}>", name))
        }
    }
}

#[napi(object)]
pub struct HoistLimits {
    pub max_ask_calls: Option<u32>,
    pub max_collection_size: Option<u32>,
    pub max_string_size: Option<u32>,
    pub max_steps: Option<u32>,
}

/// Run a Hoist program.
///
/// @param source - The Hoist source code
/// @param context - Optional context string
/// @param limits - Optional resource limits
#[napi]
pub fn run(
    env: Env,
    source: String,
    context: Option<String>,
    limits: Option<HoistLimits>,
) -> Result<napi::JsUnknown> {
    let program = parse_source(&source)
        .map_err(|e| Error::new(Status::InvalidArg, format!("Compile error: {}", e)))?;

    let mut resource_limits = ResourceLimits::default();
    if let Some(l) = limits {
        if let Some(v) = l.max_ask_calls {
            resource_limits.max_ask_calls = v as usize;
        }
        if let Some(v) = l.max_collection_size {
            resource_limits.max_collection_size = v as usize;
        }
        if let Some(v) = l.max_string_size {
            resource_limits.max_string_size = v as usize;
        }
        if let Some(v) = l.max_steps {
            resource_limits.max_steps = v as usize;
        }
    }

    let mut interp = Interpreter::new().with_limits(resource_limits);

    if let Some(ctx) = context {
        interp.set_var("context", Value::String(ctx));
    }

    let result = interp
        .eval_program(&program)
        .map_err(|e| Error::new(Status::GenericFailure, format!("{}", e)))?;

    let json_val = value_to_json(&result);
    env.to_js_value(&json_val)
}

/// Check if a Hoist program is valid (parse only).
#[napi]
pub fn check(source: String) -> Result<bool> {
    parse_source(&source)
        .map(|_| true)
        .map_err(|e| Error::new(Status::InvalidArg, format!("Compile error: {}", e)))
}
