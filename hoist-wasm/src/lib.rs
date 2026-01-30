use wasm_bindgen::prelude::*;
use hoist_core::{parse_source, Interpreter, Value, ResourceLimits};
use hoist_core::interpreter::AskHandler;

/// Convert a Hoist Value to a JsValue.
fn value_to_js(val: &Value) -> JsValue {
    match val {
        Value::Int(n) => JsValue::from(*n as f64),
        Value::Bool(b) => JsValue::from(*b),
        Value::String(s) => JsValue::from_str(s),
        Value::List(items) => {
            let arr = js_sys::Array::new();
            for item in items {
                arr.push(&value_to_js(item));
            }
            arr.into()
        }
        Value::Record(map) => {
            let obj = js_sys::Object::new();
            for (k, v) in map {
                let _ = js_sys::Reflect::set(&obj, &JsValue::from_str(k), &value_to_js(v));
            }
            obj.into()
        }
        Value::Tuple(items) => {
            let arr = js_sys::Array::new();
            for item in items {
                arr.push(&value_to_js(item));
            }
            arr.into()
        }
        Value::Optional(Some(inner)) => value_to_js(inner),
        Value::Optional(None) => JsValue::NULL,
        Value::Closure { .. } => JsValue::from_str("<function>"),
        Value::BuiltinFn(name) => JsValue::from_str(&format!("<builtin:{}>", name)),
    }
}

/// Ask handler that uses a JS callback function.
struct JsAskHandler {
    callback: js_sys::Function,
}

// WASM is single-threaded, but we need these traits for the AskHandler bound.
unsafe impl Send for JsAskHandler {}
unsafe impl Sync for JsAskHandler {}

impl AskHandler for JsAskHandler {
    fn ask(&self, prompt: &str, channel: Option<&str>) -> Result<String, String> {
        let this = JsValue::NULL;
        let prompt_js = JsValue::from_str(prompt);
        let channel_js = match channel {
            Some(ch) => JsValue::from_str(ch),
            None => JsValue::NULL,
        };
        let result = self
            .callback
            .call2(&this, &prompt_js, &channel_js)
            .map_err(|e| format!("JS ask handler error: {:?}", e))?;
        result
            .as_string()
            .ok_or_else(|| "Ask handler must return a string".to_string())
    }
}

/// The Hoist runtime for WebAssembly.
#[wasm_bindgen]
pub struct HoistRuntime {
    ask_handler: Option<js_sys::Function>,
    context: Option<String>,
    variables: Vec<(String, Value)>,
    limits: ResourceLimits,
}

#[wasm_bindgen]
impl HoistRuntime {
    /// Create a new Hoist runtime.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            ask_handler: None,
            context: None,
            variables: Vec::new(),
            limits: ResourceLimits::default(),
        }
    }

    /// Set the ask handler function: (prompt: string, channel: string|null) => string
    #[wasm_bindgen(js_name = setAskHandler)]
    pub fn set_ask_handler(&mut self, handler: js_sys::Function) {
        self.ask_handler = Some(handler);
    }

    /// Set the context variable.
    #[wasm_bindgen(js_name = setContext)]
    pub fn set_context(&mut self, context: &str) {
        self.context = Some(context.to_string());
    }

    /// Set a string variable.
    #[wasm_bindgen(js_name = setVariable)]
    pub fn set_variable(&mut self, name: &str, value: &str) {
        self.variables
            .push((name.to_string(), Value::String(value.to_string())));
    }

    /// Configure resource limits.
    #[wasm_bindgen(js_name = setLimits)]
    pub fn set_limits(
        &mut self,
        max_ask_calls: u32,
        max_collection_size: u32,
        max_string_size: u32,
        max_steps: u32,
    ) {
        self.limits = ResourceLimits {
            max_ask_calls: max_ask_calls as usize,
            max_collection_size: max_collection_size as usize,
            max_string_size: max_string_size as usize,
            max_steps: max_steps as usize,
        };
    }

    /// Run a Hoist program and return the result.
    pub fn run(&self, source: &str) -> Result<JsValue, JsValue> {
        let program = parse_source(source)
            .map_err(|e| JsValue::from_str(&format!("Compile error: {}", e)))?;

        let mut interp = Interpreter::new().with_limits(self.limits.clone());

        if let Some(ref handler) = self.ask_handler {
            interp = interp.with_ask_handler(Box::new(JsAskHandler {
                callback: handler.clone(),
            }));
        }

        if let Some(ref ctx) = self.context {
            interp.set_var("context", Value::String(ctx.clone()));
        }

        for (name, val) in &self.variables {
            interp.set_var(name, val.clone());
        }

        let result = interp
            .eval_program(&program)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?;

        Ok(value_to_js(&result))
    }

    /// Parse and check a Hoist program (returns true if valid).
    pub fn check(&self, source: &str) -> Result<bool, JsValue> {
        parse_source(source)
            .map(|_| true)
            .map_err(|e| JsValue::from_str(&format!("Compile error: {}", e)))
    }
}

/// Convenience: run a Hoist program with optional context.
#[wasm_bindgen(js_name = hoistRun)]
pub fn hoist_run(source: &str, context: Option<String>) -> Result<JsValue, JsValue> {
    let mut runtime = HoistRuntime::new();
    if let Some(ctx) = context {
        runtime.set_context(&ctx);
    }
    runtime.run(source)
}

/// Check if a Hoist program is valid.
#[wasm_bindgen(js_name = hoistCheck)]
pub fn hoist_check(source: &str) -> Result<bool, JsValue> {
    let runtime = HoistRuntime::new();
    runtime.check(source)
}
