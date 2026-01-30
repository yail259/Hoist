use std::fmt;

/// Source location in a Hoist program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub column: usize,
}

impl Span {
    pub fn new(start: usize, end: usize, line: usize, column: usize) -> Self {
        Self { start, end, line, column }
    }

    pub fn dummy() -> Self {
        Self::default()
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// All errors produced by the Hoist compiler and runtime.
#[derive(Debug, thiserror::Error)]
pub enum HoistError {
    // --- Lexer errors ---
    #[error("Syntax error at {span}: {message}")]
    SyntaxError { message: String, span: Span },

    #[error("Unterminated string literal at {span}")]
    UnterminatedString { span: Span },

    #[error("Invalid escape sequence '\\{ch}' at {span}")]
    InvalidEscape { ch: char, span: Span },

    // --- Parser errors ---
    #[error("Parse error at {span}: expected {expected}, found {found}")]
    ParseError {
        expected: String,
        found: String,
        span: Span,
    },

    #[error("Unexpected end of input{}", if let Some(s) = span { format!(" at {s}") } else { String::new() })]
    UnexpectedEof { span: Option<Span> },

    // --- Type errors ---
    #[error("Type error at {span}: {message}")]
    TypeError { message: String, span: Span },

    #[error("Unbound variable '{name}' at {span}")]
    UnboundVariable { name: String, span: Span },

    #[error("Type mismatch at {span}: expected {expected}, found {actual}")]
    TypeMismatch {
        expected: String,
        actual: String,
        span: Span,
    },

    // --- Runtime errors ---
    #[error("Runtime error: {message}")]
    RuntimeError { message: String },

    #[error("Division by zero")]
    DivisionByZero,

    #[error("Index out of bounds: index {index}, length {length}")]
    IndexOutOfBounds { index: i64, length: usize },

    #[error("Ask failed: {message}")]
    AskFailed { message: String },

    #[error("Limit exceeded: {message}")]
    LimitExceeded { message: String },

    #[error("Not callable: attempted to call a non-function value")]
    NotCallable,

    #[error("Arity mismatch: expected {expected} arguments, got {got}")]
    ArityMismatch { expected: usize, got: usize },

    #[error("Type error at runtime: {message}")]
    RuntimeTypeError { message: String },
}

pub type Result<T> = std::result::Result<T, HoistError>;
