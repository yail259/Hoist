use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::{PyAnyMethods, PyDict, PyList, PyModule, PyTuple};
use hoist_core::{parse_source, Interpreter, Value, ResourceLimits};
use hoist_core::interpreter::AskHandler;
use std::sync::Arc;

/// Convert a Hoist Value to a Python object.
fn value_to_py(py: Python<'_>, val: &Value) -> Py<PyAny> {
    match val {
        Value::Int(n) => n.into_pyobject(py).unwrap().into_any().unbind(),
        Value::Bool(b) => b.into_pyobject(py).unwrap().to_owned().into_any().unbind(),
        Value::String(s) => s.into_pyobject(py).unwrap().into_any().unbind(),
        Value::List(items) => {
            let list: Vec<Py<PyAny>> = items.iter().map(|v| value_to_py(py, v)).collect();
            list.into_pyobject(py).unwrap().into_any().unbind()
        }
        Value::Record(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                let _ = dict.set_item(k, value_to_py(py, v));
            }
            dict.into_pyobject(py).unwrap().into_any().unbind()
        }
        Value::Tuple(items) => {
            let elements: Vec<Py<PyAny>> = items.iter().map(|v| value_to_py(py, v)).collect();
            PyTuple::new(py, elements).unwrap().into_pyobject(py).unwrap().into_any().unbind()
        }
        Value::Optional(Some(inner)) => value_to_py(py, inner),
        Value::Optional(None) => py.None(),
        Value::Closure { .. } => "<function>".into_pyobject(py).unwrap().into_any().unbind(),
        Value::BuiltinFn(name) => format!("<builtin:{}>", name).into_pyobject(py).unwrap().into_any().unbind(),
    }
}

/// Convert a Python object to a Hoist Value.
fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Optional(None));
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Value::Bool(b));
    }
    if let Ok(n) = obj.extract::<i64>() {
        return Ok(Value::Int(n));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::String(s));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let items: PyResult<Vec<Value>> = list.iter().map(|item| py_to_value(&item)).collect();
        return Ok(Value::List(items?));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = std::collections::HashMap::new();
        for (k, v) in dict {
            let key: String = k.extract()?;
            map.insert(key, py_to_value(&v)?);
        }
        return Ok(Value::Record(map));
    }
    Err(PyValueError::new_err(format!(
        "Cannot convert Python type '{}' to Hoist Value",
        obj.get_type().name()?
    )))
}

/// Python-callable ask handler that wraps a Python function.
struct PyAskHandler {
    callback: Arc<Py<PyAny>>,
}

impl AskHandler for PyAskHandler {
    fn ask(&self, prompt: &str, channel: Option<&str>) -> Result<String, String> {
        Python::attach(|py| {
            let result = self
                .callback
                .call(py, (prompt, channel), None)
                .map_err(|e| format!("Python ask handler error: {}", e))?;
            result
                .extract::<String>(py)
                .map_err(|e| format!("Ask handler must return a string: {}", e))
        })
    }
}

/// The Hoist runtime for Python.
#[pyclass]
struct HoistRuntime {
    ask_handler: Option<Arc<Py<PyAny>>>,
    context: Option<String>,
    variables: Vec<(String, Value)>,
    max_ask_calls: usize,
    max_collection_size: usize,
    max_string_size: usize,
    max_steps: usize,
}

#[pymethods]
impl HoistRuntime {
    #[new]
    #[pyo3(signature = (ask_handler=None))]
    fn new(ask_handler: Option<Py<PyAny>>) -> Self {
        let defaults = ResourceLimits::default();
        Self {
            ask_handler: ask_handler.map(Arc::new),
            context: None,
            variables: Vec::new(),
            max_ask_calls: defaults.max_ask_calls,
            max_collection_size: defaults.max_collection_size,
            max_string_size: defaults.max_string_size,
            max_steps: defaults.max_steps,
        }
    }

    /// Set the context variable.
    fn set_context(&mut self, context: String) {
        self.context = Some(context);
    }

    /// Set a variable in the runtime environment.
    fn set_variable(&mut self, name: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let val = py_to_value(value)?;
        self.variables.push((name, val));
        Ok(())
    }

    /// Configure resource limits.
    #[pyo3(signature = (max_ask_calls=None, max_collection_size=None, max_string_size=None, max_steps=None))]
    fn set_limits(
        &mut self,
        max_ask_calls: Option<usize>,
        max_collection_size: Option<usize>,
        max_string_size: Option<usize>,
        max_steps: Option<usize>,
    ) {
        if let Some(v) = max_ask_calls { self.max_ask_calls = v; }
        if let Some(v) = max_collection_size { self.max_collection_size = v; }
        if let Some(v) = max_string_size { self.max_string_size = v; }
        if let Some(v) = max_steps { self.max_steps = v; }
    }

    /// Run a Hoist program and return the result as a Python object.
    fn run(&self, py: Python<'_>, source: &str) -> PyResult<Py<PyAny>> {
        let program = parse_source(source)
            .map_err(|e| PyValueError::new_err(format!("Compile error: {}", e)))?;

        let limits = ResourceLimits {
            max_ask_calls: self.max_ask_calls,
            max_collection_size: self.max_collection_size,
            max_string_size: self.max_string_size,
            max_steps: self.max_steps,
        };

        let mut interp = Interpreter::new().with_limits(limits);

        if let Some(ref handler) = self.ask_handler {
            interp = interp.with_ask_handler(Box::new(PyAskHandler {
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
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;

        Ok(value_to_py(py, &result))
    }

    /// Parse a Hoist program and check for errors (returns True if valid).
    fn check(&self, source: &str) -> PyResult<bool> {
        match parse_source(source) {
            Ok(_) => Ok(true),
            Err(e) => Err(PyValueError::new_err(format!("Compile error: {}", e))),
        }
    }
}

/// Convenience function: run a Hoist program with an optional ask handler.
#[pyfunction]
#[pyo3(signature = (source, context=None, ask_handler=None))]
fn run(
    py: Python<'_>,
    source: &str,
    context: Option<String>,
    ask_handler: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let mut runtime = HoistRuntime::new(ask_handler);
    if let Some(ctx) = context {
        runtime.set_context(ctx);
    }
    runtime.run(py, source)
}

/// Python module definition.
#[pymodule]
fn hoist(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HoistRuntime>()?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}
