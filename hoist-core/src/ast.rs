use crate::error::Span;

/// A complete Hoist program: a sequence of bindings followed by a return expression.
#[derive(Debug, Clone)]
pub struct Program {
    pub bindings: Vec<Binding>,
    pub return_expr: Expr,
}

/// A let-binding: `let name = value`
#[derive(Debug, Clone)]
pub struct Binding {
    pub name: String,
    pub type_ann: Option<TypeExpr>,
    pub value: Expr,
    pub span: Span,
}

/// An expression node with source location.
#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

impl Expr {
    pub fn new(kind: ExprKind, span: Span) -> Self {
        Self { kind, span }
    }
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    // Literals
    IntLit(i64),
    BoolLit(bool),
    StringLit(Vec<StringPart>),

    // Variable reference
    Var(String),

    // Binary operation
    BinOp {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    // Unary operation
    UnaryOp {
        op: UnaryOp,
        operand: Box<Expr>,
    },

    // if ... then ... else ...
    If {
        condition: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Box<Expr>,
    },

    // match expr with | pattern -> expr ...
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },

    // List literal [a, b, c]
    List(Vec<Expr>),

    // Record literal {field: value, ...}
    Record(Vec<(String, Expr)>),

    // Index: expr[expr]
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
    },

    // Member access: expr.field
    Member {
        base: Box<Expr>,
        field: String,
    },

    // Lambda: param -> body   or  (p1, p2) -> body
    Lambda {
        params: Vec<String>,
        body: Box<Expr>,
    },

    // Function call: func(args...)
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
    },

    // Tuple: (a, b, c)
    Tuple(Vec<Expr>),

    // --- Collection operations (first-class syntax) ---
    // map collection with transform
    Map {
        collection: Box<Expr>,
        transform: Box<Expr>,
    },

    // filter collection where predicate
    Filter {
        collection: Box<Expr>,
        predicate: Box<Expr>,
    },

    // fold collection from initial with acc, elem -> body
    Fold {
        collection: Box<Expr>,
        initial: Box<Expr>,
        acc_name: String,
        elem_name: String,
        body: Box<Expr>,
    },

    // take n from collection
    Take {
        count: Box<Expr>,
        collection: Box<Expr>,
    },

    // drop n from collection
    Drop {
        count: Box<Expr>,
        collection: Box<Expr>,
    },

    // split text by delimiter
    Split {
        text: Box<Expr>,
        delimiter: Box<Expr>,
    },

    // join items with separator
    Join {
        list: Box<Expr>,
        separator: Box<Expr>,
    },

    // window text size n stride m
    Window {
        text: Box<Expr>,
        size: Box<Expr>,
        stride: Box<Expr>,
    },

    // slice text from start to end
    Slice {
        text: Box<Expr>,
        start: Box<Expr>,
        end: Box<Expr>,
    },

    // --- Ask expression (LLM call) ---
    Ask {
        prompt: Box<Expr>,
        modifiers: AskModifiers,
    },

    // Pipeline: expr |> expr (desugared during parsing)
    Pipe {
        left: Box<Expr>,
        right: Box<Expr>,
    },
}

/// Parts of an interpolated string.
#[derive(Debug, Clone)]
pub enum StringPart {
    Literal(String),
    Interpolation(Box<Expr>),
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,    // +
    Sub,    // -
    Mul,    // *
    Div,    // /
    Mod,    // %
    Eq,     // ==
    Ne,     // !=
    Lt,     // <
    Le,     // <=
    Gt,     // >
    Ge,     // >=
    And,    // and
    Or,     // or
    Concat, // ++
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Not, // not
    Neg, // - (unary minus)
}

/// Ask expression modifiers.
#[derive(Debug, Clone, Default)]
pub struct AskModifiers {
    pub typed_output: Option<TypeExpr>,
    pub channel: Option<String>,
    pub retries: Option<u32>,
    pub fallback: Option<Box<Expr>>,
}

/// A match arm: | pattern -> expr
#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Expr,
    pub span: Span,
}

/// Pattern for match expressions.
#[derive(Debug, Clone)]
pub enum Pattern {
    Wildcard,
    Var(String),
    IntLit(i64),
    BoolLit(bool),
    StringLit(String),
    Constructor { name: String, args: Vec<Pattern> },
}

/// Type expression used in annotations and `ask ... as Type`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeExpr {
    Named(String), // String, Int, Bool
    List(Box<TypeExpr>),
    Optional(Box<TypeExpr>),
    Tuple(Vec<TypeExpr>),
    Record(Vec<(String, TypeExpr)>),
}
