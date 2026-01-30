use std::collections::HashMap;
use std::collections::HashSet;
use crate::ast::*;
use crate::error::HoistError;
use crate::stdlib;

/// Runtime value.
#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Bool(bool),
    String(String),
    List(Vec<Value>),
    Record(HashMap<String, Value>),
    Tuple(Vec<Value>),
    Optional(Option<Box<Value>>),
    Closure {
        params: Vec<String>,
        body: Expr,
        env: Env,
    },
    BuiltinFn(String),
}

impl Value {
    pub fn as_int(&self) -> Result<i64, HoistError> {
        match self {
            Value::Int(n) => Ok(*n),
            _ => Err(HoistError::RuntimeTypeError { message: format!("expected Int, got {}", self.type_name()) }),
        }
    }

    pub fn as_bool(&self) -> Result<bool, HoistError> {
        match self {
            Value::Bool(b) => Ok(*b),
            _ => Err(HoistError::RuntimeTypeError { message: format!("expected Bool, got {}", self.type_name()) }),
        }
    }

    pub fn as_string(&self) -> Result<&str, HoistError> {
        match self {
            Value::String(s) => Ok(s),
            _ => Err(HoistError::RuntimeTypeError { message: format!("expected String, got {}", self.type_name()) }),
        }
    }

    pub fn as_list(&self) -> Result<&[Value], HoistError> {
        match self {
            Value::List(l) => Ok(l),
            _ => Err(HoistError::RuntimeTypeError { message: format!("expected List, got {}", self.type_name()) }),
        }
    }

    pub fn as_sequence(&self) -> Result<&[Value], HoistError> {
        match self {
            Value::List(l) => Ok(l),
            Value::Tuple(t) => Ok(t),
            _ => Err(HoistError::RuntimeTypeError { message: format!("expected List or Tuple, got {}", self.type_name()) }),
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "Int",
            Value::Bool(_) => "Bool",
            Value::String(_) => "String",
            Value::List(_) => "List",
            Value::Record(_) => "Record",
            Value::Tuple(_) => "Tuple",
            Value::Optional(_) => "Optional",
            Value::Closure { .. } => "Function",
            Value::BuiltinFn(_) => "BuiltinFn",
        }
    }

    /// Convert value to display string (for `show` and interpolation).
    pub fn display(&self) -> String {
        match self {
            Value::Int(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::String(s) => s.clone(),
            Value::List(items) => {
                let parts: Vec<String> = items.iter().map(|v| v.display()).collect();
                format!("[{}]", parts.join(", "))
            }
            Value::Record(fields) => {
                let parts: Vec<String> = fields.iter()
                    .map(|(k, v)| format!("{}: {}", k, v.display()))
                    .collect();
                format!("{{{}}}", parts.join(", "))
            }
            Value::Tuple(items) => {
                let parts: Vec<String> = items.iter().map(|v| v.display()).collect();
                format!("({})", parts.join(", "))
            }
            Value::Optional(Some(v)) => format!("Some({})", v.display()),
            Value::Optional(None) => "None".into(),
            Value::Closure { .. } => "<function>".into(),
            Value::BuiltinFn(name) => format!("<builtin:{}>", name),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::List(a), Value::List(b)) => a == b,
            (Value::Optional(a), Value::Optional(b)) => a == b,
            _ => false,
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => a.partial_cmp(b),
            (Value::String(a), Value::String(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

pub type Env = HashMap<String, Value>;

/// Host callback for the `ask` primitive.
pub trait AskHandler: Send + Sync {
    fn ask(&self, prompt: &str, channel: Option<&str>) -> Result<String, String>;
}

/// Resource limits for execution.
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    pub max_ask_calls: usize,
    pub max_collection_size: usize,
    pub max_string_size: usize,
    pub max_steps: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_ask_calls: 100,
            max_collection_size: 10_000,
            max_string_size: 10 * 1024 * 1024, // 10 MB
            max_steps: 1_000_000,
        }
    }
}

/// The Hoist interpreter.
pub struct Interpreter {
    env: Env,
    ask_handler: Option<Box<dyn AskHandler>>,
    limits: ResourceLimits,
    ask_count: usize,
    step_count: usize,
    /// Captured ask prompts (for testing).
    pub captured_asks: Vec<String>,
}

impl Interpreter {
    pub fn new() -> Self {
        let mut env = Env::new();
        stdlib::register_builtins(&mut env);

        Self {
            env,
            ask_handler: None,
            limits: ResourceLimits::default(),
            ask_count: 0,
            step_count: 0,
            captured_asks: Vec::new(),
        }
    }

    pub fn with_ask_handler(mut self, handler: Box<dyn AskHandler>) -> Self {
        self.ask_handler = Some(handler);
        self
    }

    pub fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn set_var(&mut self, name: &str, value: Value) {
        self.env.insert(name.to_string(), value);
    }

    pub fn eval_program(&mut self, program: &Program) -> Result<Value, HoistError> {
        // Evaluate bindings (detect duplicates per spec §4.2)
        let mut bound_names: HashSet<String> = HashSet::new();
        for binding in &program.bindings {
            if bound_names.contains(&binding.name) {
                return Err(HoistError::TypeError {
                    message: format!("name '{}' is already bound in this scope", binding.name),
                    span: binding.span,
                });
            }
            let val = self.eval_expr(&binding.value)?;
            self.env.insert(binding.name.clone(), val);
            bound_names.insert(binding.name.clone());
        }

        // Evaluate return expression
        self.eval_expr(&program.return_expr)
    }

    pub fn eval_expr(&mut self, expr: &Expr) -> Result<Value, HoistError> {
        self.step_count += 1;
        if self.step_count > self.limits.max_steps {
            return Err(HoistError::LimitExceeded {
                message: format!("execution step limit ({}) exceeded", self.limits.max_steps),
            });
        }

        match &expr.kind {
            ExprKind::IntLit(n) => Ok(Value::Int(*n)),
            ExprKind::BoolLit(b) => Ok(Value::Bool(*b)),
            ExprKind::StringLit(parts) => self.eval_string_parts(parts),

            ExprKind::Var(name) => {
                if name == "None" {
                    return Ok(Value::Optional(None));
                }
                self.env.get(name).cloned().ok_or_else(|| HoistError::UnboundVariable {
                    name: name.clone(),
                    span: expr.span,
                })
            }

            ExprKind::BinOp { op, left, right } => {
                // Short-circuit for and/or
                if *op == BinOp::And {
                    let l = self.eval_expr(left)?;
                    if !l.as_bool()? { return Ok(Value::Bool(false)); }
                    return self.eval_expr(right);
                }
                if *op == BinOp::Or {
                    let l = self.eval_expr(left)?;
                    // Optional `or` unwrap: `expr or default` (spec §5.6)
                    match &l {
                        Value::Optional(Some(inner)) => return Ok((**inner).clone()),
                        Value::Optional(None) => return self.eval_expr(right),
                        Value::Bool(true) => return Ok(Value::Bool(true)),
                        Value::Bool(false) => return self.eval_expr(right),
                        _ => return Err(HoistError::RuntimeTypeError {
                            message: format!("'or' requires Bool or Optional, got {}", l.type_name()),
                        }),
                    }
                }

                let l = self.eval_expr(left)?;
                let r = self.eval_expr(right)?;
                self.eval_binop(*op, l, r)
            }

            ExprKind::UnaryOp { op, operand } => {
                let val = self.eval_expr(operand)?;
                match op {
                    UnaryOp::Not => Ok(Value::Bool(!val.as_bool()?)),
                    UnaryOp::Neg => Ok(Value::Int(-val.as_int()?)),
                }
            }

            ExprKind::If { condition, then_branch, else_branch } => {
                let cond = self.eval_expr(condition)?.as_bool()?;
                if cond {
                    self.eval_expr(then_branch)
                } else {
                    self.eval_expr(else_branch)
                }
            }

            ExprKind::Match { scrutinee, arms } => {
                let val = self.eval_expr(scrutinee)?;
                for arm in arms {
                    if let Some(bindings) = self.match_pattern(&arm.pattern, &val) {
                        let old_env = self.env.clone();
                        self.env.extend(bindings);
                        let result = self.eval_expr(&arm.body);
                        self.env = old_env;
                        return result;
                    }
                }
                Err(HoistError::RuntimeError {
                    message: "non-exhaustive match".into(),
                })
            }

            ExprKind::List(items) => {
                let vals: Result<Vec<_>, _> = items.iter()
                    .map(|e| self.eval_expr(e))
                    .collect();
                let vals = vals?;
                if vals.len() > self.limits.max_collection_size {
                    return Err(HoistError::LimitExceeded {
                        message: format!("collection size {} exceeds limit {}", vals.len(), self.limits.max_collection_size),
                    });
                }
                Ok(Value::List(vals))
            }

            ExprKind::Record(fields) => {
                let mut map = HashMap::new();
                for (name, expr) in fields {
                    map.insert(name.clone(), self.eval_expr(expr)?);
                }
                Ok(Value::Record(map))
            }

            ExprKind::Tuple(items) => {
                let vals: Result<Vec<_>, _> = items.iter()
                    .map(|e| self.eval_expr(e))
                    .collect();
                Ok(Value::Tuple(vals?))
            }

            ExprKind::Index { base, index } => {
                let base_val = self.eval_expr(base)?;
                let idx = self.eval_expr(index)?.as_int()?;

                match base_val {
                    Value::List(list) => {
                        let actual_idx = if idx < 0 {
                            (list.len() as i64 + idx) as usize
                        } else {
                            idx as usize
                        };
                        Ok(list.get(actual_idx)
                            .map(|v| Value::Optional(Some(Box::new(v.clone()))))
                            .unwrap_or(Value::Optional(None)))
                    }
                    _ => Err(HoistError::RuntimeTypeError {
                        message: "index operator requires a List".into(),
                    }),
                }
            }

            ExprKind::Member { base, field } => {
                let base_val = self.eval_expr(base)?;
                match base_val {
                    Value::Record(map) => {
                        map.get(field).cloned().ok_or_else(|| HoistError::RuntimeError {
                            message: format!("record has no field '{}'", field),
                        })
                    }
                    _ => Err(HoistError::RuntimeTypeError {
                        message: "member access requires a Record".into(),
                    }),
                }
            }

            ExprKind::Lambda { params, body } => {
                Ok(Value::Closure {
                    params: params.clone(),
                    body: (**body).clone(),
                    env: self.env.clone(),
                })
            }

            ExprKind::Call { func, args } => {
                let func_val = self.eval_expr(func)?;
                let mut arg_vals = Vec::new();
                for arg in args {
                    arg_vals.push(self.eval_expr(arg)?);
                }
                self.call_function(func_val, arg_vals)
            }

            // --- Collection operations ---
            ExprKind::Map { collection, transform } => {
                let list = self.eval_expr(collection)?;
                let items = list.as_list()?.to_vec();
                let func = self.eval_expr(transform)?;

                let mut results = Vec::new();
                for item in items {
                    results.push(self.call_function(func.clone(), vec![item])?);
                }
                self.check_list_size(results.len())?;
                Ok(Value::List(results))
            }

            ExprKind::Filter { collection, predicate } => {
                let list = self.eval_expr(collection)?;
                let items = list.as_list()?.to_vec();
                let func = self.eval_expr(predicate)?;

                let mut results = Vec::new();
                for item in items {
                    let keep = self.call_function(func.clone(), vec![item.clone()])?.as_bool()?;
                    if keep {
                        results.push(item);
                    }
                }
                self.check_list_size(results.len())?;
                Ok(Value::List(results))
            }

            ExprKind::Fold { collection, initial, acc_name, elem_name, body } => {
                let list = self.eval_expr(collection)?;
                let items = list.as_list()?.to_vec();
                let mut acc = self.eval_expr(initial)?;

                for item in items {
                    let old_env = self.env.clone();
                    self.env.insert(acc_name.clone(), acc);
                    self.env.insert(elem_name.clone(), item);
                    acc = self.eval_expr(body)?;
                    self.env = old_env;
                }
                Ok(acc)
            }

            ExprKind::Take { count, collection } => {
                let n = self.eval_expr(count)?.as_int()? as usize;
                let list = self.eval_expr(collection)?;
                let items = list.as_list()?;
                Ok(Value::List(items.iter().take(n).cloned().collect()))
            }

            ExprKind::Drop { count, collection } => {
                let n = self.eval_expr(count)?.as_int()? as usize;
                let list = self.eval_expr(collection)?;
                let items = list.as_list()?;
                Ok(Value::List(items.iter().skip(n).cloned().collect()))
            }

            ExprKind::Split { text, delimiter } => {
                let s = self.eval_expr(text)?.as_string()?.to_owned();
                let delim = self.eval_expr(delimiter)?.as_string()?.to_owned();
                let parts: Vec<Value> = s.split(&delim)
                    .map(|p| Value::String(p.to_string()))
                    .collect();
                self.check_list_size(parts.len())?;
                Ok(Value::List(parts))
            }

            ExprKind::Join { list, separator } => {
                let items = self.eval_expr(list)?;
                let items = items.as_list()?.to_vec();
                let sep = self.eval_expr(separator)?.as_string()?.to_owned();
                let strings: Result<Vec<String>, _> = items.iter()
                    .map(|v| match v {
                        Value::String(s) => Ok(s.clone()),
                        other => Ok(other.display()),
                    })
                    .collect();
                let result = strings?.join(&sep);
                self.check_string_size(&result)?;
                Ok(Value::String(result))
            }

            ExprKind::Window { text, size, stride } => {
                let s = self.eval_expr(text)?.as_string()?.to_owned();
                let win_size = self.eval_expr(size)?.as_int()? as usize;
                let win_stride = self.eval_expr(stride)?.as_int()? as usize;

                let chars: Vec<char> = s.chars().collect();
                let mut windows = Vec::new();
                let mut i = 0;
                while i < chars.len() {
                    let end = (i + win_size).min(chars.len());
                    let chunk: String = chars[i..end].iter().collect();
                    windows.push(Value::String(chunk));
                    i += win_stride;
                    if i + win_stride > chars.len() && end == chars.len() {
                        break;
                    }
                }
                self.check_list_size(windows.len())?;
                Ok(Value::List(windows))
            }

            ExprKind::Slice { text, start, end } => {
                let s = self.eval_expr(text)?.as_string()?.to_owned();
                let chars: Vec<char> = s.chars().collect();
                let len = chars.len() as i64;

                let mut s_idx = self.eval_expr(start)?.as_int()?;
                let mut e_idx = self.eval_expr(end)?.as_int()?;

                // Handle negative indices
                if s_idx < 0 { s_idx += len; }
                if e_idx < 0 { e_idx += len; }

                let s_idx = s_idx.max(0) as usize;
                let e_idx = (e_idx as usize).min(chars.len());

                if s_idx >= e_idx {
                    Ok(Value::String(String::new()))
                } else {
                    Ok(Value::String(chars[s_idx..e_idx].iter().collect()))
                }
            }

            // --- Ask ---
            ExprKind::Ask { prompt, modifiers } => {
                self.eval_ask(prompt, modifiers)
            }

            ExprKind::Pipe { left, right } => {
                // Should be desugared during parsing, but handle just in case
                let _l = self.eval_expr(left)?;
                self.eval_expr(right)
            }
        }
    }

    fn eval_ask(&mut self, prompt_expr: &Expr, modifiers: &AskModifiers) -> Result<Value, HoistError> {
        self.ask_count += 1;
        if self.ask_count > self.limits.max_ask_calls {
            return Err(HoistError::LimitExceeded {
                message: format!("ask call limit ({}) exceeded", self.limits.max_ask_calls),
            });
        }

        let prompt = self.eval_expr(prompt_expr)?;
        let prompt_str = prompt.as_string()?.to_owned();

        // Augment prompt with type instruction if `as Type` is specified
        let final_prompt = if let Some(ref type_expr) = modifiers.typed_output {
            format!("{}\n\n{}", prompt_str, type_instruction(type_expr))
        } else {
            prompt_str.clone()
        };

        self.captured_asks.push(prompt_str);

        let handler = self.ask_handler.as_ref().ok_or_else(|| HoistError::AskFailed {
            message: "no ask handler configured".into(),
        })?;

        let max_retries = modifiers.retries.unwrap_or(0);
        let channel = modifiers.channel.as_deref();

        for attempt in 0..=max_retries {
            match handler.ask(&final_prompt, channel) {
                Ok(response) => {
                    // Parse typed output if `as Type` is specified
                    if let Some(ref type_expr) = modifiers.typed_output {
                        match parse_typed_response(&response, type_expr) {
                            Ok(val) => return Ok(val),
                            Err(_) if attempt < max_retries => continue, // retry on parse failure
                            Err(e) => {
                                if let Some(fallback) = &modifiers.fallback {
                                    return self.eval_expr(fallback);
                                }
                                return Err(HoistError::AskFailed {
                                    message: format!("typed output parsing failed: {}", e),
                                });
                            }
                        }
                    }
                    return Ok(Value::String(response));
                }
                Err(e) if attempt < max_retries => {
                    continue; // retry
                }
                Err(e) => {
                    // Try fallback
                    if let Some(fallback) = &modifiers.fallback {
                        return self.eval_expr(fallback);
                    }
                    return Err(HoistError::AskFailed { message: e });
                }
            }
        }

        unreachable!()
    }

    fn eval_string_parts(&mut self, parts: &[StringPart]) -> Result<Value, HoistError> {
        let mut result = String::new();
        for part in parts {
            match part {
                StringPart::Literal(s) => result.push_str(s),
                StringPart::Interpolation(expr) => {
                    let val = self.eval_expr(expr)?;
                    result.push_str(&val.display());
                }
            }
        }
        self.check_string_size(&result)?;
        Ok(Value::String(result))
    }

    fn eval_binop(&self, op: BinOp, left: Value, right: Value) -> Result<Value, HoistError> {
        match op {
            BinOp::Add => match (&left, &right) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
                (Value::String(a), Value::String(b)) => {
                    let result = format!("{}{}", a, b);
                    self.check_string_size(&result)?;
                    Ok(Value::String(result))
                }
                _ => Err(HoistError::RuntimeTypeError {
                    message: format!("cannot add {} and {}", left.type_name(), right.type_name()),
                }),
            },
            BinOp::Sub => Ok(Value::Int(left.as_int()? - right.as_int()?)),
            BinOp::Mul => Ok(Value::Int(left.as_int()? * right.as_int()?)),
            BinOp::Div => {
                let r = right.as_int()?;
                if r == 0 { return Err(HoistError::DivisionByZero); }
                Ok(Value::Int(left.as_int()? / r))
            }
            BinOp::Mod => {
                let r = right.as_int()?;
                if r == 0 { return Err(HoistError::DivisionByZero); }
                Ok(Value::Int(left.as_int()? % r))
            }
            BinOp::Eq => Ok(Value::Bool(left == right)),
            BinOp::Ne => Ok(Value::Bool(left != right)),
            BinOp::Lt => Ok(Value::Bool(left < right)),
            BinOp::Le => Ok(Value::Bool(left <= right)),
            BinOp::Gt => Ok(Value::Bool(left > right)),
            BinOp::Ge => Ok(Value::Bool(left >= right)),
            BinOp::And | BinOp::Or => unreachable!("handled in eval_expr"),
            BinOp::Concat => {
                match (&left, &right) {
                    (Value::String(a), Value::String(b)) => {
                        let result = format!("{}{}", a, b);
                        self.check_string_size(&result)?;
                        Ok(Value::String(result))
                    }
                    (Value::List(a), Value::List(b)) => {
                        let mut result = a.clone();
                        result.extend(b.clone());
                        self.check_list_size(result.len())?;
                        Ok(Value::List(result))
                    }
                    _ => Err(HoistError::RuntimeTypeError {
                        message: format!("cannot concat {} and {}", left.type_name(), right.type_name()),
                    }),
                }
            }
        }
    }

    fn call_function(&mut self, func: Value, args: Vec<Value>) -> Result<Value, HoistError> {
        match func {
            Value::Closure { params, body, env } => {
                if params.len() != args.len() {
                    return Err(HoistError::ArityMismatch {
                        expected: params.len(),
                        got: args.len(),
                    });
                }

                let old_env = std::mem::replace(&mut self.env, env);
                for (param, arg) in params.iter().zip(args) {
                    self.env.insert(param.clone(), arg);
                }

                let result = self.eval_expr(&body);
                self.env = old_env;
                result
            }
            Value::BuiltinFn(name) => {
                stdlib::call_builtin(self, &name, args)
            }
            _ => Err(HoistError::NotCallable),
        }
    }

    fn match_pattern(&self, pattern: &Pattern, value: &Value) -> Option<Env> {
        match (pattern, value) {
            (Pattern::Wildcard, _) => Some(HashMap::new()),

            (Pattern::Var(name), val) => {
                let mut env = HashMap::new();
                env.insert(name.clone(), val.clone());
                Some(env)
            }

            (Pattern::IntLit(n), Value::Int(v)) if n == v => Some(HashMap::new()),
            (Pattern::BoolLit(b), Value::Bool(v)) if b == v => Some(HashMap::new()),
            (Pattern::StringLit(s), Value::String(v)) if s == v => Some(HashMap::new()),

            (Pattern::Constructor { name, args: pats }, _) => {
                match (name.as_str(), value) {
                    ("Some", Value::Optional(Some(inner))) if pats.len() == 1 => {
                        self.match_pattern(&pats[0], inner)
                    }
                    ("None", Value::Optional(None)) if pats.is_empty() => {
                        Some(HashMap::new())
                    }
                    _ => None,
                }
            }

            _ => None,
        }
    }

    fn check_string_size(&self, s: &str) -> Result<(), HoistError> {
        if s.len() > self.limits.max_string_size {
            Err(HoistError::LimitExceeded {
                message: format!(
                    "string size {} bytes exceeds limit {} bytes",
                    s.len(),
                    self.limits.max_string_size
                ),
            })
        } else {
            Ok(())
        }
    }

    fn check_list_size(&self, len: usize) -> Result<(), HoistError> {
        if len > self.limits.max_collection_size {
            Err(HoistError::LimitExceeded {
                message: format!(
                    "collection size {} exceeds limit {}",
                    len, self.limits.max_collection_size
                ),
            })
        } else {
            Ok(())
        }
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------
// Typed ask output support
// ---------------------------------------------------------------

/// Generate a prompt instruction suffix for the expected type.
fn type_instruction(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Named(name) => match name.as_str() {
            "String" => "Respond with plain text.".into(),
            "Int" => "Respond with a single integer (no other text).".into(),
            "Bool" => "Respond with exactly \"true\" or \"false\" (no other text).".into(),
            other => format!("Respond with a value of type {}.", other),
        },
        TypeExpr::List(inner) => {
            format!(
                "Respond with a JSON array of {}. Output only valid JSON, no other text.",
                type_description(inner)
            )
        }
        TypeExpr::Optional(inner) => {
            format!(
                "Respond with {} or \"null\" if not applicable. Output only valid JSON, no other text.",
                type_description(inner)
            )
        }
        TypeExpr::Tuple(items) => {
            let descs: Vec<String> = items.iter().map(type_description).collect();
            format!(
                "Respond with a JSON array of exactly {} elements: [{}]. Output only valid JSON, no other text.",
                items.len(),
                descs.join(", ")
            )
        }
        TypeExpr::Record(fields) => {
            let field_descs: Vec<String> = fields
                .iter()
                .map(|(name, ty)| format!("\"{}\" ({})", name, type_description(ty)))
                .collect();
            format!(
                "Respond with a JSON object with fields: {}. Output only valid JSON, no other text.",
                field_descs.join(", ")
            )
        }
    }
}

fn type_description(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Named(name) => match name.as_str() {
            "String" => "strings".into(),
            "Int" => "integers".into(),
            "Bool" => "booleans".into(),
            other => other.to_string(),
        },
        TypeExpr::List(inner) => format!("arrays of {}", type_description(inner)),
        TypeExpr::Optional(inner) => format!("nullable {}", type_description(inner)),
        TypeExpr::Tuple(items) => {
            let parts: Vec<String> = items.iter().map(type_description).collect();
            format!("tuples of ({})", parts.join(", "))
        }
        TypeExpr::Record(fields) => {
            let parts: Vec<String> = fields.iter().map(|(n, t)| format!("{}: {}", n, type_description(t))).collect();
            format!("objects with {{{}}}", parts.join(", "))
        }
    }
}

/// Parse a response string into a typed Value according to the expected TypeExpr.
fn parse_typed_response(response: &str, ty: &TypeExpr) -> Result<Value, String> {
    let trimmed = response.trim();
    match ty {
        TypeExpr::Named(name) => match name.as_str() {
            "String" => Ok(Value::String(trimmed.to_string())),
            "Int" => trimmed
                .parse::<i64>()
                .map(Value::Int)
                .map_err(|_| format!("expected integer, got: {}", truncate(trimmed, 50))),
            "Bool" => match trimmed.to_lowercase().as_str() {
                "true" => Ok(Value::Bool(true)),
                "false" => Ok(Value::Bool(false)),
                _ => Err(format!("expected true/false, got: {}", truncate(trimmed, 50))),
            },
            _ => Ok(Value::String(trimmed.to_string())),
        },
        TypeExpr::List(inner) => {
            let json: serde_json::Value = serde_json::from_str(trimmed)
                .map_err(|e| format!("invalid JSON: {}", e))?;
            let arr = json
                .as_array()
                .ok_or_else(|| "expected JSON array".to_string())?;
            let mut values = Vec::new();
            for item in arr {
                values.push(json_item_to_typed_value(item, inner)?);
            }
            Ok(Value::List(values))
        }
        TypeExpr::Optional(inner) => {
            if trimmed == "null" || trimmed.is_empty() {
                Ok(Value::Optional(None))
            } else {
                let val = parse_typed_response(trimmed, inner)?;
                Ok(Value::Optional(Some(Box::new(val))))
            }
        }
        TypeExpr::Tuple(items) => {
            let json: serde_json::Value = serde_json::from_str(trimmed)
                .map_err(|e| format!("invalid JSON: {}", e))?;
            let arr = json
                .as_array()
                .ok_or_else(|| "expected JSON array for tuple".to_string())?;
            if arr.len() != items.len() {
                return Err(format!(
                    "expected tuple of {} elements, got {}",
                    items.len(),
                    arr.len()
                ));
            }
            let mut values = Vec::new();
            for (item, ty) in arr.iter().zip(items) {
                values.push(json_item_to_typed_value(item, ty)?);
            }
            Ok(Value::Tuple(values))
        }
        TypeExpr::Record(fields) => {
            let json: serde_json::Value = serde_json::from_str(trimmed)
                .map_err(|e| format!("invalid JSON: {}", e))?;
            let obj = json
                .as_object()
                .ok_or_else(|| "expected JSON object".to_string())?;
            let mut record = HashMap::new();
            for (name, ty) in fields {
                let val = obj
                    .get(name)
                    .ok_or_else(|| format!("missing field '{}'", name))?;
                record.insert(name.clone(), json_item_to_typed_value(val, ty)?);
            }
            Ok(Value::Record(record))
        }
    }
}

/// Convert a serde_json::Value into a typed Hoist Value.
fn json_item_to_typed_value(json: &serde_json::Value, ty: &TypeExpr) -> Result<Value, String> {
    match ty {
        TypeExpr::Named(name) => match name.as_str() {
            "String" => json
                .as_str()
                .map(|s| Value::String(s.to_string()))
                .ok_or_else(|| format!("expected string, got: {}", json)),
            "Int" => json
                .as_i64()
                .map(Value::Int)
                .ok_or_else(|| format!("expected integer, got: {}", json)),
            "Bool" => json
                .as_bool()
                .map(Value::Bool)
                .ok_or_else(|| format!("expected boolean, got: {}", json)),
            _ => Ok(stdlib::json_to_value(json.clone())),
        },
        TypeExpr::List(inner) => {
            let arr = json
                .as_array()
                .ok_or_else(|| format!("expected array, got: {}", json))?;
            let mut values = Vec::new();
            for item in arr {
                values.push(json_item_to_typed_value(item, inner)?);
            }
            Ok(Value::List(values))
        }
        TypeExpr::Optional(inner) => {
            if json.is_null() {
                Ok(Value::Optional(None))
            } else {
                let val = json_item_to_typed_value(json, inner)?;
                Ok(Value::Optional(Some(Box::new(val))))
            }
        }
        TypeExpr::Tuple(items) => {
            let arr = json
                .as_array()
                .ok_or_else(|| format!("expected array for tuple, got: {}", json))?;
            if arr.len() != items.len() {
                return Err(format!("tuple length mismatch: expected {}, got {}", items.len(), arr.len()));
            }
            let mut values = Vec::new();
            for (item, ty) in arr.iter().zip(items) {
                values.push(json_item_to_typed_value(item, ty)?);
            }
            Ok(Value::Tuple(values))
        }
        TypeExpr::Record(fields) => {
            let obj = json
                .as_object()
                .ok_or_else(|| format!("expected object, got: {}", json))?;
            let mut record = HashMap::new();
            for (name, ty) in fields {
                let val = obj
                    .get(name)
                    .ok_or_else(|| format!("missing field '{}'", name))?;
                record.insert(name.clone(), json_item_to_typed_value(val, ty)?);
            }
            Ok(Value::Record(record))
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_source;

    fn eval(src: &str) -> Value {
        let prog = parse_source(src).expect("parse failed");
        let mut interp = Interpreter::new();
        interp.eval_program(&prog).expect("eval failed")
    }

    fn eval_with_context(src: &str, ctx: Vec<(&str, Value)>) -> Value {
        let prog = parse_source(src).expect("parse failed");
        let mut interp = Interpreter::new();
        for (name, val) in ctx {
            interp.set_var(name, val);
        }
        interp.eval_program(&prog).expect("eval failed")
    }

    #[test]
    fn test_int_literal() {
        assert_eq!(eval("return 42"), Value::Int(42));
    }

    #[test]
    fn test_bool_literal() {
        assert_eq!(eval("return true"), Value::Bool(true));
    }

    #[test]
    fn test_string_literal() {
        assert_eq!(eval(r#"return "hello""#), Value::String("hello".into()));
    }

    #[test]
    fn test_arithmetic() {
        assert_eq!(eval("return 2 + 3 * 4"), Value::Int(14));
        assert_eq!(eval("return 10 - 3"), Value::Int(7));
        assert_eq!(eval("return 10 / 3"), Value::Int(3));
        assert_eq!(eval("return 10 % 3"), Value::Int(1));
    }

    #[test]
    fn test_comparison() {
        assert_eq!(eval("return 1 == 1"), Value::Bool(true));
        assert_eq!(eval("return 1 != 2"), Value::Bool(true));
        assert_eq!(eval("return 1 < 2"), Value::Bool(true));
        assert_eq!(eval("return 2 <= 2"), Value::Bool(true));
        assert_eq!(eval("return 3 > 2"), Value::Bool(true));
    }

    #[test]
    fn test_boolean_ops() {
        assert_eq!(eval("return true and false"), Value::Bool(false));
        assert_eq!(eval("return true or false"), Value::Bool(true));
        assert_eq!(eval("return not true"), Value::Bool(false));
    }

    #[test]
    fn test_let_binding() {
        assert_eq!(eval("let x = 42\nreturn x"), Value::Int(42));
    }

    #[test]
    fn test_if_then_else() {
        assert_eq!(eval("return if true then 1 else 2"), Value::Int(1));
        assert_eq!(eval("return if false then 1 else 2"), Value::Int(2));
    }

    #[test]
    fn test_string_interpolation() {
        assert_eq!(
            eval(r#"let name = "world"
return "hello {name}!""#),
            Value::String("hello world!".into())
        );
    }

    #[test]
    fn test_list() {
        let result = eval("return [1, 2, 3]");
        match result {
            Value::List(items) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], Value::Int(1));
            }
            _ => panic!("expected list"),
        }
    }

    #[test]
    fn test_map() {
        let result = eval("let items = [1, 2, 3]\nreturn map items with it + 1");
        assert_eq!(result, Value::List(vec![Value::Int(2), Value::Int(3), Value::Int(4)]));
    }

    #[test]
    fn test_filter() {
        let result = eval("let items = [1, 2, 3, 4, 5]\nreturn filter items where it > 3");
        assert_eq!(result, Value::List(vec![Value::Int(4), Value::Int(5)]));
    }

    #[test]
    fn test_fold() {
        let result = eval("let nums = [1, 2, 3, 4]\nreturn fold nums from 0 with acc, n -> acc + n");
        assert_eq!(result, Value::Int(10));
    }

    #[test]
    fn test_split_join() {
        let result = eval(r#"let parts = split "a,b,c" by ","
return join parts with " ""#);
        assert_eq!(result, Value::String("a b c".into()));
    }

    #[test]
    fn test_take_drop() {
        let result = eval("return take 2 from [1, 2, 3, 4]");
        assert_eq!(result, Value::List(vec![Value::Int(1), Value::Int(2)]));

        let result = eval("return drop 2 from [1, 2, 3, 4]");
        assert_eq!(result, Value::List(vec![Value::Int(3), Value::Int(4)]));
    }

    #[test]
    fn test_stdlib_length() {
        assert_eq!(eval(r#"return length("hello")"#), Value::Int(5));
        assert_eq!(eval("return length([1, 2, 3])"), Value::Int(3));
    }

    #[test]
    fn test_stdlib_upper_lower() {
        assert_eq!(eval(r#"return upper("hello")"#), Value::String("HELLO".into()));
        assert_eq!(eval(r#"return lower("HELLO")"#), Value::String("hello".into()));
    }

    #[test]
    fn test_stdlib_contains() {
        assert_eq!(eval(r#"return contains("hello world", "world")"#), Value::Bool(true));
        assert_eq!(eval(r#"return contains("hello world", "xyz")"#), Value::Bool(false));
    }

    #[test]
    fn test_function_call() {
        assert_eq!(eval(r#"return trim("  hello  ")"#), Value::String("hello".into()));
    }

    #[test]
    fn test_concat_operator() {
        assert_eq!(eval(r#"return "hello" ++ " " ++ "world""#), Value::String("hello world".into()));
    }

    #[test]
    fn test_list_concat() {
        let result = eval("return [1, 2] ++ [3, 4]");
        assert_eq!(result, Value::List(vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4)]));
    }

    #[test]
    fn test_pipeline() {
        let result = eval(r#"let items = [1, 2, 3, 4, 5]
return items |> filter where it > 2 |> map with it * 10"#);
        assert_eq!(result, Value::List(vec![Value::Int(30), Value::Int(40), Value::Int(50)]));
    }

    #[test]
    fn test_match_optional() {
        let result = eval(r#"let x = Some(42)
return match x with
| Some(v) -> v
| None -> 0"#);
        // Indexing returns Optional, Some constructor for now we test match with direct var
        assert_eq!(result, Value::Int(42));
    }

    #[test]
    fn test_record() {
        let result = eval(r#"let r = {name: "Alice", age: 30}
return r.name"#);
        assert_eq!(result, Value::String("Alice".into()));
    }

    #[test]
    fn test_slice() {
        assert_eq!(
            eval(r#"return slice "hello world" from 0 to 5"#),
            Value::String("hello".into())
        );
    }

    #[test]
    fn test_negative_index() {
        assert_eq!(
            eval(r#"return slice "hello" from 0 to -1"#),
            Value::String("hell".into())
        );
    }

    #[test]
    fn test_division_by_zero() {
        let prog = parse_source("return 1 / 0").unwrap();
        let mut interp = Interpreter::new();
        let result = interp.eval_program(&prog);
        assert!(matches!(result, Err(HoistError::DivisionByZero)));
    }

    #[test]
    fn test_ask_limit() {
        // Create a program that makes many ask calls
        let src = r#"let items = split "1,2,3,4,5,6,7,8,9,10,11" by ","
return map items with ask "X""#;
        let prog = parse_source(src).unwrap();

        struct TestHandler;
        impl AskHandler for TestHandler {
            fn ask(&self, _prompt: &str, _channel: Option<&str>) -> Result<String, String> {
                Ok("ok".into())
            }
        }

        let mut interp = Interpreter::new()
            .with_ask_handler(Box::new(TestHandler))
            .with_limits(ResourceLimits { max_ask_calls: 10, ..Default::default() });

        let result = interp.eval_program(&prog);
        assert!(matches!(result, Err(HoistError::LimitExceeded { .. })));
    }

    #[test]
    fn test_show_function() {
        assert_eq!(eval(r#"return show(42)"#), Value::String("42".into()));
        assert_eq!(eval(r#"return show(true)"#), Value::String("true".into()));
    }

    #[test]
    fn test_parse_int() {
        let result = eval(r#"return parse_int("42")"#);
        assert_eq!(result, Value::Optional(Some(Box::new(Value::Int(42)))));
    }

    #[test]
    fn test_lines_words() {
        let result = eval(r#"return lines("a\nb\nc")"#);
        assert_eq!(result, Value::List(vec![
            Value::String("a".into()),
            Value::String("b".into()),
            Value::String("c".into()),
        ]));
    }

    #[test]
    fn test_context_variable() {
        let result = eval_with_context(
            r#"return split context by ",""#,
            vec![("context", Value::String("a,b,c".into()))],
        );
        assert_eq!(result, Value::List(vec![
            Value::String("a".into()),
            Value::String("b".into()),
            Value::String("c".into()),
        ]));
    }

    #[test]
    fn test_explicit_lambda() {
        let result = eval("let items = [1, 2, 3]\nreturn map items with x -> x * 2");
        assert_eq!(result, Value::List(vec![Value::Int(2), Value::Int(4), Value::Int(6)]));
    }

    #[test]
    fn test_nested_collection_ops() {
        let result = eval(r#"let items = split "1,2,3,4,5" by ","
let filtered = filter items where length(it) > 0
return show(length(filtered))"#);
        assert_eq!(result, Value::String("5".into()));
    }
}
