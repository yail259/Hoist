# Hoist Implementation Guide

A practical guide for implementing the Hoist language specification.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Lexer Implementation](#2-lexer-implementation)
3. [Parser Implementation](#3-parser-implementation)
4. [Abstract Syntax Tree](#4-abstract-syntax-tree)
5. [Type System Implementation](#5-type-system-implementation)
6. [Intermediate Representation](#6-intermediate-representation)
7. [Interpreter](#7-interpreter)
8. [Host Integration](#8-host-integration)
9. [Standard Library](#9-standard-library)
10. [Language Bindings](#10-language-bindings)
11. [Testing Strategy](#11-testing-strategy)
12. [Performance Considerations](#12-performance-considerations)
13. [Reference Implementation Roadmap](#13-reference-implementation-roadmap)

---

## 1. Architecture Overview

### Compilation Pipeline

```
Source Code
    │
    ▼
┌─────────┐
│  Lexer  │  → Token Stream
└────┬────┘
     │
     ▼
┌─────────┐
│ Parser  │  → Concrete Syntax Tree (CST)
└────┬────┘
     │
     ▼
┌─────────┐
│AST Build│  → Abstract Syntax Tree (AST)
└────┬────┘
     │
     ▼
┌─────────┐
│Type Check│ → Typed AST + Type Errors
└────┬────┘
     │
     ▼
┌─────────┐
│IR Lower │  → Intermediate Representation
└────┬────┘
     │
     ▼
┌─────────┐
│Optimizer│  → Optimized IR (optional)
└────┬────┘
     │
     ▼
┌─────────┐
│Interpret│  → Result / Host Callbacks
└─────────┘
```

### Recommended Technology Stack

| Component | Rust | Python | TypeScript |
|-----------|------|--------|------------|
| Lexer | logos | lark | moo |
| Parser | pest / lalrpop | lark | nearley / chevrotain |
| Regex | regex (RE2-compatible) | re2 | re2-wasm |
| JSON | serde_json | json (stdlib) | native |
| Bindings | - | PyO3 | napi-rs / wasm-bindgen |

### Core Crate Structure (Rust)

```
hoist/
├── Cargo.toml
├── hoist-core/           # Core library
│   ├── src/
│   │   ├── lib.rs
│   │   ├── lexer.rs
│   │   ├── parser.rs
│   │   ├── ast.rs
│   │   ├── types.rs
│   │   ├── typecheck.rs
│   │   ├── ir.rs
│   │   ├── interpreter.rs
│   │   ├── stdlib.rs
│   │   └── error.rs
│   └── Cargo.toml
├── hoist-cli/            # Command-line tool
├── hoist-python/         # Python bindings (PyO3)
├── hoist-node/           # Node.js bindings (napi-rs)
├── hoist-wasm/           # WebAssembly target
└── tests/
    └── compliance/       # JSON test suite
```

---

## 2. Lexer Implementation

### Token Types

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Keywords
    Let,
    Return,
    If,
    Then,
    Else,
    Match,
    With,
    Where,
    By,
    And,
    Or,
    Not,
    True,
    False,
    Null,
    As,           // Typed output
    Via,          // Named channels
    Retries,      // Retry clause
    Fallback,     // Fallback clause

    // Identifiers and Literals
    Ident(String),
    Int(i64),
    Float(f64),
    String(String),       // Interpolation handled in parser

    // Operators
    Plus,         // +
    Minus,        // -
    Star,         // *
    Slash,        // /
    Percent,      // %
    Eq,           // ==
    Ne,           // !=
    Lt,           // <
    Le,           // <=
    Gt,           // >
    Ge,           // >=
    Pipe,         // |>
    Dot,          // .
    DotDot,       // ..
    Colon,        // :
    Comma,        // ,
    Arrow,        // ->

    // Delimiters
    LParen,       // (
    RParen,       // )
    LBracket,     // [
    RBracket,     // ]
    LBrace,       // {
    RBrace,       // }

    // Special
    Newline,
    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

#[derive(Debug, Clone, Copy)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub column: usize,
}
```

### Lexer with Logos (Rust)

```rust
use logos::Logos;

#[derive(Logos, Debug, PartialEq, Clone)]
pub enum Token {
    // Keywords (must come before Ident)
    #[token("let")]
    Let,
    #[token("return")]
    Return,
    #[token("if")]
    If,
    #[token("then")]
    Then,
    #[token("else")]
    Else,
    #[token("match")]
    Match,
    #[token("with")]
    With,
    #[token("where")]
    Where,
    #[token("by")]
    By,
    #[token("and")]
    And,
    #[token("or")]
    Or,
    #[token("not")]
    Not,
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("null")]
    Null,

    // Identifiers
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),

    // Numbers
    #[regex(r"-?[0-9]+\.[0-9]+", |lex| lex.slice().parse().ok())]
    Float(f64),
    #[regex(r"-?[0-9]+", |lex| lex.slice().parse().ok())]
    Int(i64),

    // Strings (simplified - real impl needs interpolation)
    #[regex(r#""[^"]*""#, |lex| {
        let s = lex.slice();
        s[1..s.len()-1].to_string()
    })]
    String(String),

    // Operators
    #[token("|>")]
    Pipe,
    #[token("==")]
    Eq,
    #[token("!=")]
    Ne,
    #[token("<=")]
    Le,
    #[token(">=")]
    Ge,
    #[token("->")]
    Arrow,
    #[token("..")]
    DotDot,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token(".")]
    Dot,
    #[token(":")]
    Colon,
    #[token(",")]
    Comma,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,

    // Whitespace and comments
    #[regex(r"[ \t]+", logos::skip)]
    #[regex(r"#[^\n]*", logos::skip)]
    Whitespace,

    #[regex(r"\n+")]
    Newline,

    #[error]
    Error,
}
```

### String Interpolation

String interpolation requires special handling. The lexer should emit a sequence of tokens:

```
"Hello {name}, you have {count} items"
```

Becomes:

```
StringStart("Hello ")
LBrace
Ident("name")
RBrace
StringMiddle(", you have ")
LBrace
Ident("count")
RBrace
StringEnd(" items")
```

Implementation approach:

```rust
fn lex_string(&mut self) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_interpolation = false;
    let mut brace_depth = 0;

    self.advance(); // consume opening quote

    while let Some(c) = self.current_char() {
        if in_interpolation {
            if c == '{' {
                brace_depth += 1;
                // continue lexing expression
            } else if c == '}' {
                if brace_depth == 0 {
                    in_interpolation = false;
                    tokens.push(Token::RBrace);
                } else {
                    brace_depth -= 1;
                }
            } else {
                // Lex the interpolated expression normally
                tokens.extend(self.lex_expression_until('}'));
            }
        } else {
            match c {
                '"' => {
                    if !current.is_empty() {
                        tokens.push(Token::StringPart(current.clone()));
                    }
                    self.advance();
                    break;
                }
                '{' => {
                    if !current.is_empty() {
                        tokens.push(Token::StringPart(current.clone()));
                        current.clear();
                    }
                    tokens.push(Token::LBrace);
                    in_interpolation = true;
                    self.advance();
                }
                '\\' => {
                    self.advance();
                    current.push(self.escape_char()?);
                }
                _ => {
                    current.push(c);
                    self.advance();
                }
            }
        }
    }

    tokens
}
```

---

## 3. Parser Implementation

### Grammar (pest format)

```pest
// hoist.pest - PEG grammar for pest parser

program = { SOI ~ statement* ~ EOI }

statement = { let_stmt | return_stmt | expr }

let_stmt = { "let" ~ ident ~ "=" ~ expr }
return_stmt = { "return" ~ expr }

expr = { pipeline }

pipeline = { logical_or ~ ("|>" ~ pipeline_call)* }
pipeline_call = { ident ~ pipeline_args? }
pipeline_args = { expr ~ ("," ~ expr)* }

logical_or = { logical_and ~ ("or" ~ logical_and)* }
logical_and = { comparison ~ ("and" ~ comparison)* }

comparison = { additive ~ (comp_op ~ additive)? }
comp_op = { "==" | "!=" | "<=" | ">=" | "<" | ">" }

additive = { multiplicative ~ (("+"|"-") ~ multiplicative)* }
multiplicative = { unary ~ (("*"|"/"|"%") ~ unary)* }

unary = { "not" ~ unary | postfix }

postfix = { primary ~ postfix_op* }
postfix_op = { member_access | index_access | call }
member_access = { "." ~ ident }
index_access = { "[" ~ expr ~ "]" }
call = { "(" ~ (expr ~ ("," ~ expr)*)? ~ ")" }

primary = {
    if_expr |
    match_expr |
    lambda |
    list_literal |
    record_literal |
    "(" ~ expr ~ ")" |
    literal |
    ident
}

if_expr = { "if" ~ expr ~ "then" ~ expr ~ "else" ~ expr }

match_expr = { "match" ~ expr ~ "{" ~ match_arm+ ~ "}" }
match_arm = { pattern ~ "->" ~ expr ~ ","? }

lambda = {
    ident ~ "->" ~ expr |
    "(" ~ (ident ~ ("," ~ ident)*)? ~ ")" ~ "->" ~ expr
}

list_literal = { "[" ~ (expr ~ ("," ~ expr)*)? ~ "]" }
record_literal = { "{" ~ (field ~ ("," ~ field)*)? ~ "}" }
field = { ident ~ ":" ~ expr }

pattern = {
    "_" |
    literal |
    ident ~ "(" ~ (pattern ~ ("," ~ pattern)*)? ~ ")" |
    ident
}

literal = { float | int | string | "true" | "false" | "null" }

// Ask expression with AI-friendly modifiers
ask_expr = { "ask" ~ expr ~ ask_modifier* }
ask_modifier = {
    "as" ~ type_spec |
    "via" ~ ident |
    "with" ~ "retries" ~ ":" ~ int |
    "fallback" ~ expr
}

type_spec = {
    "String" | "Int" | "Bool" |
    "List" ~ "<" ~ type_spec ~ ">" |
    "{" ~ (ident ~ ":" ~ type_spec ~ ("," ~ ident ~ ":" ~ type_spec)*)? ~ "}"
}

ident = @{ (ASCII_ALPHA | "_") ~ (ASCII_ALPHANUMERIC | "_")* }
int = @{ "-"? ~ ASCII_DIGIT+ }
float = @{ "-"? ~ ASCII_DIGIT+ ~ "." ~ ASCII_DIGIT+ }
string = @{ "\"" ~ (!"\"" ~ ANY)* ~ "\"" }

WHITESPACE = _{ " " | "\t" | "\n" | "\r" }
COMMENT = _{ "#" ~ (!"\n" ~ ANY)* }
```

### Recursive Descent Parser (Rust)

```rust
pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn parse(&mut self) -> Result<Program, ParseError> {
        let mut statements = Vec::new();

        while !self.at_end() {
            if self.check(TokenKind::Newline) {
                self.advance();
                continue;
            }
            statements.push(self.parse_statement()?);
        }

        Ok(Program { statements })
    }

    fn parse_statement(&mut self) -> Result<Statement, ParseError> {
        if self.check(TokenKind::Let) {
            self.parse_let()
        } else if self.check(TokenKind::Return) {
            self.parse_return()
        } else {
            Ok(Statement::Expr(self.parse_expr()?))
        }
    }

    fn parse_let(&mut self) -> Result<Statement, ParseError> {
        self.expect(TokenKind::Let)?;
        let name = self.parse_ident()?;
        self.expect(TokenKind::Eq)?;
        let value = self.parse_expr()?;
        Ok(Statement::Let { name, value })
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_pipeline()
    }

    fn parse_pipeline(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_logical_or()?;

        while self.match_token(TokenKind::Pipe) {
            let func = self.parse_ident()?;

            // Check for additional arguments
            let mut args = vec![expr];
            if !self.check_pipeline_end() {
                // Parse with/where/by clauses or direct arguments
                args.extend(self.parse_pipeline_args()?);
            }

            expr = Expr::Call {
                func: Box::new(Expr::Var(func)),
                args,
            };
        }

        Ok(expr)
    }

    fn parse_pipeline_args(&mut self) -> Result<Vec<Expr>, ParseError> {
        let mut args = Vec::new();

        // Handle special keywords
        if self.match_token(TokenKind::With) {
            args.push(self.parse_expr()?);
        } else if self.match_token(TokenKind::Where) {
            args.push(self.parse_lambda_or_expr()?);
        } else if self.match_token(TokenKind::By) {
            args.push(self.parse_expr()?);
        } else {
            // Direct arguments
            args.push(self.parse_expr()?);
            while self.match_token(TokenKind::Comma) {
                args.push(self.parse_expr()?);
            }
        }

        Ok(args)
    }

    fn parse_logical_or(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_logical_and()?;

        while self.match_token(TokenKind::Or) {
            let right = self.parse_logical_and()?;
            left = Expr::BinOp {
                op: BinOp::Or,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    // ... similar for other precedence levels

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        if self.match_token(TokenKind::If) {
            self.parse_if()
        } else if self.match_token(TokenKind::Match) {
            self.parse_match()
        } else if self.match_token(TokenKind::LBracket) {
            self.parse_list()
        } else if self.match_token(TokenKind::LBrace) {
            self.parse_record()
        } else if self.match_token(TokenKind::LParen) {
            let expr = self.parse_expr()?;
            self.expect(TokenKind::RParen)?;
            Ok(expr)
        } else if let Some(lit) = self.parse_literal()? {
            Ok(Expr::Literal(lit))
        } else {
            let name = self.parse_ident()?;

            // Check for lambda
            if self.match_token(TokenKind::Arrow) {
                let body = self.parse_expr()?;
                Ok(Expr::Lambda {
                    params: vec![name],
                    body: Box::new(body),
                })
            } else {
                Ok(Expr::Var(name))
            }
        }
    }
}
```

### Handling `it` Implicit Parameter

The `it` keyword is syntactic sugar. When parsing expressions in `where` clauses, wrap in a lambda if `it` is referenced:

```rust
fn parse_where_clause(&mut self) -> Result<Expr, ParseError> {
    let expr = self.parse_expr()?;

    if self.references_it(&expr) {
        // Wrap in implicit lambda
        Ok(Expr::Lambda {
            params: vec!["it".to_string()],
            body: Box::new(expr),
        })
    } else {
        Ok(expr)
    }
}

fn references_it(&self, expr: &Expr) -> bool {
    match expr {
        Expr::Var(name) => name == "it",
        Expr::BinOp { left, right, .. } => {
            self.references_it(left) || self.references_it(right)
        }
        Expr::Call { func, args } => {
            self.references_it(func) || args.iter().any(|a| self.references_it(a))
        }
        // ... other cases
        _ => false,
    }
}
```

---

## 4. Abstract Syntax Tree

### AST Node Definitions

```rust
#[derive(Debug, Clone)]
pub struct Program {
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub enum Statement {
    Let {
        name: String,
        type_ann: Option<Type>,
        value: Expr,
    },
    Return(Expr),
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub enum Expr {
    // Literals
    Literal(Literal),

    // Variables
    Var(String),

    // Operations
    BinOp {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    UnaryOp {
        op: UnaryOp,
        operand: Box<Expr>,
    },

    // Control flow
    If {
        condition: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Box<Expr>,
    },
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },

    // Data structures
    List(Vec<Expr>),
    Record(Vec<(String, Expr)>),

    // Access
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
    },
    Member {
        base: Box<Expr>,
        field: String,
    },

    // Functions
    Lambda {
        params: Vec<String>,
        body: Box<Expr>,
    },
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
    },

    // LLM Integration (AI-friendly features)
    Ask {
        prompt: Box<Expr>,
        modifiers: AskModifiers,
    },
}

/// AI-friendly ask modifiers for structured output, routing, and error handling
#[derive(Debug, Clone, Default)]
pub struct AskModifiers {
    /// Expected output type (e.g., List<String>, {name: String, age: Int})
    pub typed_output: Option<TypeSpec>,
    /// Named channel for routing to different LLM configurations
    pub channel: Option<String>,
    /// Number of retry attempts on failure
    pub retries: Option<u32>,
    /// Fallback expression if ask fails
    pub fallback: Option<Box<Expr>>,
}

/// Type specification for typed ask output
#[derive(Debug, Clone)]
pub enum TypeSpec {
    String,
    Int,
    Bool,
    List(Box<TypeSpec>),
    Record(Vec<(String, TypeSpec)>),
}

#[derive(Debug, Clone)]
pub enum Literal {
    Int(i64),
    Float(f64),
    String(StringLiteral),
    Bool(bool),
    Null,
}

#[derive(Debug, Clone)]
pub struct StringLiteral {
    pub parts: Vec<StringPart>,
}

#[derive(Debug, Clone)]
pub enum StringPart {
    Text(String),
    Interpolation(Box<Expr>),
}

#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    Eq, Ne, Lt, Le, Gt, Ge,
    And, Or,
    Concat,
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Not,
    Neg,
}

#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Expr,
}

#[derive(Debug, Clone)]
pub enum Pattern {
    Wildcard,
    Literal(Literal),
    Var(String),
    Constructor {
        name: String,
        args: Vec<Pattern>,
    },
}
```

### Typed AST

After type checking, the AST is annotated with types:

```rust
#[derive(Debug, Clone)]
pub struct TypedExpr {
    pub expr: Expr,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct TypedProgram {
    pub statements: Vec<TypedStatement>,
    pub return_type: Type,
}
```

---

## 5. Type System Implementation

### Type Representation

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    // Primitives
    Int,
    Float,
    String,
    Bool,
    Null,

    // Compound
    List(Box<Type>),
    Record(Vec<(String, Type)>),

    // Functions
    Function {
        params: Vec<Type>,
        ret: Box<Type>,
    },

    // Type variables (for inference)
    Var(TypeVar),

    // Union types
    Union(Vec<Type>),

    // Any (top type, use sparingly)
    Any,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeVar(pub u32);
```

### Type Checker

```rust
pub struct TypeChecker {
    env: HashMap<String, Type>,
    substitutions: HashMap<TypeVar, Type>,
    next_var: u32,
    errors: Vec<TypeError>,
}

impl TypeChecker {
    pub fn check_program(&mut self, program: &Program) -> Result<TypedProgram, Vec<TypeError>> {
        let mut typed_stmts = Vec::new();

        for stmt in &program.statements {
            typed_stmts.push(self.check_statement(stmt)?);
        }

        // Find return type from last return statement
        let return_type = self.find_return_type(&typed_stmts);

        if self.errors.is_empty() {
            Ok(TypedProgram {
                statements: typed_stmts,
                return_type,
            })
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    fn check_expr(&mut self, expr: &Expr) -> Result<TypedExpr, TypeError> {
        match expr {
            Expr::Literal(lit) => {
                let ty = self.literal_type(lit);
                Ok(TypedExpr { expr: expr.clone(), ty, span: Span::dummy() })
            }

            Expr::Var(name) => {
                let ty = self.env.get(name)
                    .cloned()
                    .ok_or_else(|| TypeError::UndefinedVar(name.clone()))?;
                Ok(TypedExpr { expr: expr.clone(), ty, span: Span::dummy() })
            }

            Expr::BinOp { op, left, right } => {
                let left_typed = self.check_expr(left)?;
                let right_typed = self.check_expr(right)?;
                let result_ty = self.check_binop(*op, &left_typed.ty, &right_typed.ty)?;

                Ok(TypedExpr {
                    expr: expr.clone(),
                    ty: result_ty,
                    span: Span::dummy(),
                })
            }

            Expr::Call { func, args } => {
                let func_typed = self.check_expr(func)?;
                let mut arg_types = Vec::new();

                for arg in args {
                    arg_types.push(self.check_expr(arg)?);
                }

                let ret_ty = self.check_call(&func_typed.ty, &arg_types)?;

                Ok(TypedExpr {
                    expr: expr.clone(),
                    ty: ret_ty,
                    span: Span::dummy(),
                })
            }

            Expr::Lambda { params, body } => {
                // Create fresh type variables for parameters
                let mut param_types = Vec::new();
                let old_env = self.env.clone();

                for param in params {
                    let ty = Type::Var(self.fresh_var());
                    self.env.insert(param.clone(), ty.clone());
                    param_types.push(ty);
                }

                let body_typed = self.check_expr(body)?;
                self.env = old_env;

                Ok(TypedExpr {
                    expr: expr.clone(),
                    ty: Type::Function {
                        params: param_types,
                        ret: Box::new(body_typed.ty),
                    },
                    span: Span::dummy(),
                })
            }

            // ... other cases
        }
    }

    fn unify(&mut self, t1: &Type, t2: &Type) -> Result<Type, TypeError> {
        let t1 = self.apply_substitutions(t1);
        let t2 = self.apply_substitutions(t2);

        match (&t1, &t2) {
            (Type::Var(v), t) | (t, Type::Var(v)) => {
                if !self.occurs_in(*v, t) {
                    self.substitutions.insert(*v, t.clone());
                    Ok(t.clone())
                } else {
                    Err(TypeError::InfiniteType)
                }
            }

            (Type::Int, Type::Int) => Ok(Type::Int),
            (Type::Float, Type::Float) => Ok(Type::Float),
            (Type::String, Type::String) => Ok(Type::String),
            (Type::Bool, Type::Bool) => Ok(Type::Bool),

            (Type::List(a), Type::List(b)) => {
                let elem = self.unify(a, b)?;
                Ok(Type::List(Box::new(elem)))
            }

            (Type::Function { params: p1, ret: r1 },
             Type::Function { params: p2, ret: r2 }) => {
                if p1.len() != p2.len() {
                    return Err(TypeError::ArityMismatch);
                }

                let mut params = Vec::new();
                for (a, b) in p1.iter().zip(p2.iter()) {
                    params.push(self.unify(a, b)?);
                }

                let ret = self.unify(r1, r2)?;
                Ok(Type::Function { params, ret: Box::new(ret) })
            }

            _ => Err(TypeError::Mismatch(t1, t2)),
        }
    }

    fn fresh_var(&mut self) -> TypeVar {
        let v = TypeVar(self.next_var);
        self.next_var += 1;
        v
    }
}
```

### Stdlib Type Signatures

```rust
fn init_stdlib_types(checker: &mut TypeChecker) {
    // String functions
    checker.env.insert("len".into(), Type::Function {
        params: vec![Type::String],
        ret: Box::new(Type::Int),
    });

    checker.env.insert("upper".into(), Type::Function {
        params: vec![Type::String],
        ret: Box::new(Type::String),
    });

    // List functions (polymorphic - use type variables)
    let a = checker.fresh_var();
    checker.env.insert("map".into(), Type::Function {
        params: vec![
            Type::List(Box::new(Type::Var(a))),
            Type::Function {
                params: vec![Type::Var(a)],
                ret: Box::new(Type::Var(checker.fresh_var())),
            },
        ],
        ret: Box::new(Type::List(Box::new(Type::Var(checker.fresh_var())))),
    });

    // ask function (returns String)
    checker.env.insert("ask".into(), Type::Function {
        params: vec![Type::String],
        ret: Box::new(Type::String),
    });
}
```

---

## 6. Intermediate Representation

### IR Design

The IR is a simplified, linearized form suitable for interpretation:

```rust
#[derive(Debug, Clone)]
pub enum IR {
    // Constants
    LoadConst(Value),

    // Variables
    LoadVar(String),
    StoreVar(String),

    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Neg,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Logic
    And,
    Or,
    Not,

    // String
    Concat,

    // Control flow
    Jump(usize),
    JumpIfFalse(usize),

    // Functions
    Call(usize),  // arity
    Return,

    // Data structures
    MakeList(usize),       // size
    MakeRecord(Vec<String>),  // field names
    Index,
    Member(String),

    // Built-ins
    CallBuiltin(String, usize),

    // Host interaction - AI-friendly ask with modifiers
    Ask(AskIR),
}

/// IR representation of ask with all AI-friendly modifiers
#[derive(Debug, Clone)]
pub struct AskIR {
    /// Expected output type for automatic parsing
    pub typed_output: Option<TypeSpec>,
    /// Named channel for routing
    pub channel: Option<String>,
    /// Number of retries on failure
    pub retries: u32,
    /// Index of fallback expression in constants (if any)
    pub fallback_idx: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct IRProgram {
    pub instructions: Vec<IR>,
    pub constants: Vec<Value>,
}
```

### Lowering AST to IR

```rust
pub struct IRCompiler {
    instructions: Vec<IR>,
    constants: Vec<Value>,
}

impl IRCompiler {
    pub fn compile(&mut self, program: &TypedProgram) -> IRProgram {
        for stmt in &program.statements {
            self.compile_statement(stmt);
        }

        IRProgram {
            instructions: std::mem::take(&mut self.instructions),
            constants: std::mem::take(&mut self.constants),
        }
    }

    fn compile_expr(&mut self, expr: &TypedExpr) {
        match &expr.expr {
            Expr::Literal(lit) => {
                let idx = self.add_constant(lit.clone().into());
                self.emit(IR::LoadConst(idx));
            }

            Expr::Var(name) => {
                self.emit(IR::LoadVar(name.clone()));
            }

            Expr::BinOp { op, left, right } => {
                self.compile_expr(left);
                self.compile_expr(right);
                self.emit(self.binop_to_ir(*op));
            }

            Expr::If { condition, then_branch, else_branch } => {
                self.compile_expr(condition);
                let jump_else = self.emit_placeholder();

                self.compile_expr(then_branch);
                let jump_end = self.emit_placeholder();

                self.patch_jump(jump_else, self.current_offset());
                self.compile_expr(else_branch);

                self.patch_jump(jump_end, self.current_offset());
            }

            Expr::Call { func, args } => {
                // Compile arguments
                for arg in args {
                    self.compile_expr(arg);
                }

                // Check if it's a builtin
                if let Expr::Var(name) = &func.expr {
                    if self.is_builtin(name) {
                        self.emit(IR::CallBuiltin(name.clone(), args.len()));
                        return;
                    }
                    if name == "ask" {
                        self.emit(IR::Ask);
                        return;
                    }
                }

                // Regular function call
                self.compile_expr(func);
                self.emit(IR::Call(args.len()));
            }

            // ... other cases
        }
    }
}
```

---

## 7. Interpreter

### Value Representation

```rust
#[derive(Debug, Clone)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    List(Vec<Value>),
    Record(HashMap<String, Value>),
    Closure {
        params: Vec<String>,
        body: Box<Expr>,
        env: Env,
    },
    BuiltinFn(String),
}

pub type Env = HashMap<String, Value>;
```

### Tree-Walking Interpreter

```rust
pub struct Interpreter {
    env: Env,
    host: Box<dyn HostCallbacks>,
    recursion_depth: usize,
    max_recursion: usize,
    iteration_count: usize,
    max_iterations: usize,
}

impl Interpreter {
    pub fn new(host: Box<dyn HostCallbacks>) -> Self {
        let mut env = Env::new();
        init_stdlib(&mut env);

        Self {
            env,
            host,
            recursion_depth: 0,
            max_recursion: 100,
            iteration_count: 0,
            max_iterations: 10_000,
        }
    }

    pub fn eval(&mut self, program: &Program) -> Result<Value, RuntimeError> {
        let mut result = Value::Null;

        for stmt in &program.statements {
            match stmt {
                Statement::Let { name, value, .. } => {
                    let val = self.eval_expr(value)?;
                    self.env.insert(name.clone(), val);
                }
                Statement::Return(expr) => {
                    return self.eval_expr(expr);
                }
                Statement::Expr(expr) => {
                    result = self.eval_expr(expr)?;
                }
            }
        }

        Ok(result)
    }

    fn eval_expr(&mut self, expr: &Expr) -> Result<Value, RuntimeError> {
        // Check limits
        self.iteration_count += 1;
        if self.iteration_count > self.max_iterations {
            return Err(RuntimeError::IterationLimit);
        }

        match expr {
            Expr::Literal(lit) => Ok(self.literal_to_value(lit)),

            Expr::Var(name) => {
                self.env.get(name)
                    .cloned()
                    .ok_or_else(|| RuntimeError::UndefinedVar(name.clone()))
            }

            Expr::BinOp { op, left, right } => {
                let l = self.eval_expr(left)?;

                // Short-circuit for and/or
                match op {
                    BinOp::And => {
                        if !l.as_bool()? {
                            return Ok(Value::Bool(false));
                        }
                        return self.eval_expr(right);
                    }
                    BinOp::Or => {
                        if l.as_bool()? {
                            return Ok(Value::Bool(true));
                        }
                        return self.eval_expr(right);
                    }
                    _ => {}
                }

                let r = self.eval_expr(right)?;
                self.eval_binop(*op, l, r)
            }

            Expr::If { condition, then_branch, else_branch } => {
                let cond = self.eval_expr(condition)?;
                if cond.as_bool()? {
                    self.eval_expr(then_branch)
                } else {
                    self.eval_expr(else_branch)
                }
            }

            Expr::Lambda { params, body } => {
                Ok(Value::Closure {
                    params: params.clone(),
                    body: body.clone(),
                    env: self.env.clone(),
                })
            }

            Expr::Call { func, args } => {
                let func_val = self.eval_expr(func)?;
                let mut arg_vals = Vec::new();
                for arg in args {
                    arg_vals.push(self.eval_expr(arg)?);
                }
                self.call_function(func_val, arg_vals)
            }

            Expr::List(items) => {
                let vals: Result<Vec<_>, _> = items.iter()
                    .map(|e| self.eval_expr(e))
                    .collect();
                Ok(Value::List(vals?))
            }

            // ... other cases
        }
    }

    fn call_function(&mut self, func: Value, args: Vec<Value>) -> Result<Value, RuntimeError> {
        self.recursion_depth += 1;
        if self.recursion_depth > self.max_recursion {
            return Err(RuntimeError::RecursionLimit);
        }

        let result = match func {
            Value::Closure { params, body, env } => {
                if params.len() != args.len() {
                    return Err(RuntimeError::ArityMismatch {
                        expected: params.len(),
                        got: args.len(),
                    });
                }

                // Create new environment
                let old_env = std::mem::replace(&mut self.env, env);
                for (param, arg) in params.iter().zip(args) {
                    self.env.insert(param.clone(), arg);
                }

                let result = self.eval_expr(&body);
                self.env = old_env;
                result
            }

            Value::BuiltinFn(name) => {
                self.call_builtin(&name, args)
            }

            _ => Err(RuntimeError::NotCallable),
        };

        self.recursion_depth -= 1;
        result
    }
}
```

### Host Callback Interface

```rust
pub trait HostCallbacks: Send + Sync {
    /// Called when `ask` is invoked with a prompt
    fn ask(&self, request: AskRequest) -> Result<AskResponse, HostError>;

    /// Called for logging/debugging
    fn log(&self, level: LogLevel, message: &str);

    /// Called to check if execution should be cancelled
    fn should_cancel(&self) -> bool;
}

/// Request structure for AI-friendly ask with all modifiers
#[derive(Debug, Clone)]
pub struct AskRequest {
    /// The prompt string
    pub prompt: String,
    /// Named channel for routing (None = "default")
    pub channel: Option<String>,
    /// Expected output type for parsing hints
    pub expected_type: Option<TypeSpec>,
    /// Which retry attempt this is (0 = first try)
    pub retry_count: u32,
}

/// Response from ask callback
#[derive(Debug, Clone)]
pub struct AskResponse {
    /// Raw string response from LLM
    pub raw: String,
    /// Parsed value (if typed output requested and parsing succeeded)
    pub parsed: Option<Value>,
    /// Optional metadata
    pub metadata: Option<AskMetadata>,
}

#[derive(Debug, Clone)]
pub struct AskMetadata {
    pub model: Option<String>,
    pub tokens_used: Option<u32>,
    pub latency_ms: Option<u64>,
}

pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

pub struct HostError {
    pub message: String,
    pub retryable: bool,
}
```

### Implementing AI-Friendly Ask Evaluation

```rust
impl Interpreter {
    fn eval_ask(&mut self, ask: &AskExpr) -> Result<Value, RuntimeError> {
        // Evaluate the prompt expression
        let prompt = self.eval_expr(&ask.prompt)?.as_string()?;

        // Build the request with modifiers
        let mut request = AskRequest {
            prompt: prompt.clone(),
            channel: ask.modifiers.channel.clone(),
            expected_type: ask.modifiers.typed_output.clone(),
            retry_count: 0,
        };

        // Apply typed output formatting to prompt
        if let Some(ref type_spec) = ask.modifiers.typed_output {
            request.prompt = self.format_typed_prompt(&prompt, type_spec);
        }

        // Retry loop
        let max_retries = ask.modifiers.retries.unwrap_or(0);
        let mut last_error = None;

        for attempt in 0..=max_retries {
            request.retry_count = attempt;

            match self.host.ask(request.clone()) {
                Ok(response) => {
                    // If typed output, use parsed value or parse raw
                    if let Some(ref type_spec) = ask.modifiers.typed_output {
                        if let Some(parsed) = response.parsed {
                            return Ok(parsed);
                        }
                        // Try to parse the raw response
                        match self.parse_typed_response(&response.raw, type_spec) {
                            Ok(value) => return Ok(value),
                            Err(e) if attempt < max_retries => {
                                // Parsing failed, will retry with clarification
                                request.prompt = self.format_retry_prompt(
                                    &prompt, type_spec, &e.to_string()
                                );
                                last_error = Some(e);
                                continue;
                            }
                            Err(e) => {
                                last_error = Some(e);
                                break;
                            }
                        }
                    }
                    return Ok(Value::String(response.raw));
                }
                Err(e) if e.retryable && attempt < max_retries => {
                    last_error = Some(RuntimeError::HostError(e.message.clone()));
                    continue;
                }
                Err(e) => {
                    last_error = Some(RuntimeError::HostError(e.message));
                    break;
                }
            }
        }

        // All retries exhausted, try fallback
        if let Some(ref fallback) = ask.modifiers.fallback {
            return self.eval_expr(fallback);
        }

        // No fallback, return the error
        Err(last_error.unwrap_or(RuntimeError::AskFailed("Unknown error".into())))
    }

    fn format_typed_prompt(&self, prompt: &str, type_spec: &TypeSpec) -> String {
        let format_hint = match type_spec {
            TypeSpec::String => return prompt.to_string(),
            TypeSpec::Int => "Respond with only an integer number.",
            TypeSpec::Bool => "Respond with only 'true' or 'false'.",
            TypeSpec::List(inner) => {
                return format!(
                    "{}\n\nRespond with a JSON array of {}.",
                    prompt,
                    self.type_description(inner)
                );
            }
            TypeSpec::Record(fields) => {
                let field_desc: Vec<String> = fields.iter()
                    .map(|(name, ty)| format!("\"{}\": {}", name, self.type_description(ty)))
                    .collect();
                return format!(
                    "{}\n\nRespond with only valid JSON matching this structure: {{{}}}",
                    prompt,
                    field_desc.join(", ")
                );
            }
        };
        format!("{}\n\n{}", prompt, format_hint)
    }

    fn type_description(&self, type_spec: &TypeSpec) -> &'static str {
        match type_spec {
            TypeSpec::String => "string",
            TypeSpec::Int => "integer",
            TypeSpec::Bool => "boolean",
            TypeSpec::List(_) => "array",
            TypeSpec::Record(_) => "object",
        }
    }

    fn parse_typed_response(&self, raw: &str, type_spec: &TypeSpec) -> Result<Value, RuntimeError> {
        match type_spec {
            TypeSpec::String => Ok(Value::String(raw.to_string())),
            TypeSpec::Int => {
                raw.trim().parse::<i64>()
                    .map(Value::Int)
                    .map_err(|_| RuntimeError::TypedOutputParseFailed(
                        format!("Expected integer, got: {}", raw)
                    ))
            }
            TypeSpec::Bool => {
                match raw.trim().to_lowercase().as_str() {
                    "true" => Ok(Value::Bool(true)),
                    "false" => Ok(Value::Bool(false)),
                    _ => Err(RuntimeError::TypedOutputParseFailed(
                        format!("Expected true/false, got: {}", raw)
                    ))
                }
            }
            TypeSpec::List(inner) => {
                let json: serde_json::Value = serde_json::from_str(raw.trim())
                    .map_err(|e| RuntimeError::TypedOutputParseFailed(e.to_string()))?;

                if let serde_json::Value::Array(arr) = json {
                    let values: Result<Vec<Value>, _> = arr.iter()
                        .map(|v| self.json_to_typed_value(v, inner))
                        .collect();
                    Ok(Value::List(values?))
                } else {
                    Err(RuntimeError::TypedOutputParseFailed("Expected JSON array".into()))
                }
            }
            TypeSpec::Record(fields) => {
                let json: serde_json::Value = serde_json::from_str(raw.trim())
                    .map_err(|e| RuntimeError::TypedOutputParseFailed(e.to_string()))?;

                if let serde_json::Value::Object(obj) = json {
                    let mut record = HashMap::new();
                    for (name, ty) in fields {
                        let value = obj.get(name)
                            .ok_or_else(|| RuntimeError::TypedOutputParseFailed(
                                format!("Missing field: {}", name)
                            ))?;
                        record.insert(name.clone(), self.json_to_typed_value(value, ty)?);
                    }
                    Ok(Value::Record(record))
                } else {
                    Err(RuntimeError::TypedOutputParseFailed("Expected JSON object".into()))
                }
            }
        }
    }
}
```

---

## 8. Host Integration

### Python Integration (PyO3)

```rust
// hoist-python/src/lib.rs
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass]
struct HoistRuntime {
    interpreter: Interpreter,
}

#[pymethods]
impl HoistRuntime {
    #[new]
    fn new(ask_callback: PyObject) -> PyResult<Self> {
        let host = Box::new(PyHostCallbacks { ask_callback });
        Ok(Self {
            interpreter: Interpreter::new(host),
        })
    }

    fn eval(&mut self, py: Python, source: &str, context: &PyDict) -> PyResult<PyObject> {
        // Parse
        let program = parse(source)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PySyntaxError, _>(e.to_string()))?;

        // Type check
        let typed = typecheck(&program)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("{:?}", e)))?;

        // Set context variables
        for (key, value) in context.iter() {
            let name: String = key.extract()?;
            let val = py_to_hoist(value)?;
            self.interpreter.set_var(&name, val);
        }

        // Execute
        let result = self.interpreter.eval(&program)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        hoist_to_py(py, result)
    }
}

struct PyHostCallbacks {
    ask_callback: PyObject,
}

impl HostCallbacks for PyHostCallbacks {
    fn ask(&self, prompt: &str) -> Result<String, HostError> {
        Python::with_gil(|py| {
            let result = self.ask_callback
                .call1(py, (prompt,))
                .map_err(|e| HostError {
                    message: e.to_string(),
                    retryable: false,
                })?;

            result.extract::<String>(py)
                .map_err(|e| HostError {
                    message: e.to_string(),
                    retryable: false,
                })
        })
    }

    fn log(&self, _level: LogLevel, _message: &str) {}
    fn should_cancel(&self) -> bool { false }
}

#[pymodule]
fn hoist(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<HoistRuntime>()?;
    Ok(())
}
```

### Python Usage Example

```python
import hoist
from openai import OpenAI

client = OpenAI()

def ask_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )
    return response.choices[0].message.content

# Create runtime with LLM callback
runtime = hoist.HoistRuntime(ask_callback=ask_llm)

# Hoist program
program = """
let chunks = split context by "\n\n"
let relevant = filter chunks where contains(it, "important")
let summaries = map relevant with (chunk -> ask "Summarize: {chunk}")
return join summaries with "\n---\n"
"""

# Execute with context
result = runtime.eval(program, {"context": my_long_document})
print(result)
```

### Node.js Integration (napi-rs)

```rust
// hoist-node/src/lib.rs
use napi::*;
use napi_derive::napi;

#[napi]
pub struct HoistRuntime {
    interpreter: Interpreter,
}

#[napi]
impl HoistRuntime {
    #[napi(constructor)]
    pub fn new(ask_callback: JsFunction) -> Result<Self> {
        let host = Box::new(NodeHostCallbacks {
            ask_callback: ask_callback.create_threadsafe_function(0, |ctx| {
                Ok(vec![ctx.value])
            })?,
        });

        Ok(Self {
            interpreter: Interpreter::new(host),
        })
    }

    #[napi]
    pub fn eval(&mut self, source: String, context: Object) -> Result<JsUnknown> {
        // Implementation similar to Python
        todo!()
    }
}
```

### WebAssembly Build

```rust
// hoist-wasm/src/lib.rs
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmRuntime {
    interpreter: Interpreter,
}

#[wasm_bindgen]
impl WasmRuntime {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            interpreter: Interpreter::new(Box::new(WasmHostCallbacks {})),
        }
    }

    #[wasm_bindgen]
    pub fn eval(&mut self, source: &str, context: JsValue) -> Result<JsValue, JsError> {
        // Parse and execute
        todo!()
    }

    #[wasm_bindgen]
    pub fn set_ask_callback(&mut self, callback: js_sys::Function) {
        // Set the callback for ask operations
        todo!()
    }
}
```

---

## 9. Standard Library

### Implementation Pattern

```rust
pub fn init_stdlib(env: &mut Env) {
    // String functions
    env.insert("len".into(), Value::BuiltinFn("len".into()));
    env.insert("upper".into(), Value::BuiltinFn("upper".into()));
    env.insert("lower".into(), Value::BuiltinFn("lower".into()));
    env.insert("trim".into(), Value::BuiltinFn("trim".into()));
    env.insert("split".into(), Value::BuiltinFn("split".into()));
    env.insert("join".into(), Value::BuiltinFn("join".into()));
    env.insert("contains".into(), Value::BuiltinFn("contains".into()));
    env.insert("replace".into(), Value::BuiltinFn("replace".into()));
    env.insert("matches".into(), Value::BuiltinFn("matches".into()));
    env.insert("substr".into(), Value::BuiltinFn("substr".into()));

    // List functions
    env.insert("map".into(), Value::BuiltinFn("map".into()));
    env.insert("filter".into(), Value::BuiltinFn("filter".into()));
    env.insert("reduce".into(), Value::BuiltinFn("reduce".into()));
    env.insert("first".into(), Value::BuiltinFn("first".into()));
    env.insert("last".into(), Value::BuiltinFn("last".into()));
    env.insert("take".into(), Value::BuiltinFn("take".into()));
    env.insert("drop".into(), Value::BuiltinFn("drop".into()));
    env.insert("sort".into(), Value::BuiltinFn("sort".into()));
    env.insert("unique".into(), Value::BuiltinFn("unique".into()));
    env.insert("flatten".into(), Value::BuiltinFn("flatten".into()));
    env.insert("zip".into(), Value::BuiltinFn("zip".into()));
    env.insert("range".into(), Value::BuiltinFn("range".into()));

    // Type conversion
    env.insert("int".into(), Value::BuiltinFn("int".into()));
    env.insert("float".into(), Value::BuiltinFn("float".into()));
    env.insert("str".into(), Value::BuiltinFn("str".into()));
    env.insert("bool".into(), Value::BuiltinFn("bool".into()));

    // JSON
    env.insert("parse_json".into(), Value::BuiltinFn("parse_json".into()));
    env.insert("to_json".into(), Value::BuiltinFn("to_json".into()));
}

impl Interpreter {
    fn call_builtin(&mut self, name: &str, args: Vec<Value>) -> Result<Value, RuntimeError> {
        match name {
            "len" => {
                check_arity(name, &args, 1)?;
                match &args[0] {
                    Value::String(s) => Ok(Value::Int(s.len() as i64)),
                    Value::List(l) => Ok(Value::Int(l.len() as i64)),
                    _ => Err(RuntimeError::TypeError("len requires string or list".into())),
                }
            }

            "upper" => {
                check_arity(name, &args, 1)?;
                let s = args[0].as_string()?;
                Ok(Value::String(s.to_uppercase()))
            }

            "split" => {
                check_arity(name, &args, 2)?;
                let s = args[0].as_string()?;
                let delim = args[1].as_string()?;
                let parts: Vec<Value> = s.split(&delim)
                    .map(|p| Value::String(p.to_string()))
                    .collect();
                Ok(Value::List(parts))
            }

            "map" => {
                check_arity(name, &args, 2)?;
                let list = args[0].as_list()?;
                let func = args[1].clone();

                let mut results = Vec::new();
                for item in list {
                    results.push(self.call_function(func.clone(), vec![item])?);
                }
                Ok(Value::List(results))
            }

            "filter" => {
                check_arity(name, &args, 2)?;
                let list = args[0].as_list()?;
                let predicate = args[1].clone();

                let mut results = Vec::new();
                for item in list {
                    let keep = self.call_function(predicate.clone(), vec![item.clone()])?;
                    if keep.as_bool()? {
                        results.push(item);
                    }
                }
                Ok(Value::List(results))
            }

            "reduce" => {
                check_arity(name, &args, 3)?;
                let list = args[0].as_list()?;
                let initial = args[1].clone();
                let reducer = args[2].clone();

                let mut acc = initial;
                for item in list {
                    acc = self.call_function(reducer.clone(), vec![acc, item])?;
                }
                Ok(acc)
            }

            "matches" => {
                check_arity(name, &args, 2)?;
                let s = args[0].as_string()?;
                let pattern = args[1].as_string()?;

                // Use RE2-compatible regex
                let re = regex::Regex::new(&pattern)
                    .map_err(|e| RuntimeError::RegexError(e.to_string()))?;

                let matches: Vec<Value> = re.find_iter(&s)
                    .map(|m| Value::String(m.as_str().to_string()))
                    .collect();

                Ok(Value::List(matches))
            }

            "parse_json" => {
                check_arity(name, &args, 1)?;
                let s = args[0].as_string()?;
                let json: serde_json::Value = serde_json::from_str(&s)
                    .map_err(|e| RuntimeError::JsonError(e.to_string()))?;
                Ok(json_to_value(json))
            }

            "ask" => {
                check_arity(name, &args, 1)?;
                let prompt = args[0].as_string()?;
                let response = self.host.ask(&prompt)
                    .map_err(|e| RuntimeError::HostError(e.message))?;
                Ok(Value::String(response))
            }

            _ => Err(RuntimeError::UnknownBuiltin(name.to_string())),
        }
    }
}
```

---

## 10. Language Bindings

### Binding Generation Strategy

```
┌─────────────────────────────────────────────────────────┐
│                     Rust Core                            │
│  (hoist-core: lexer, parser, typechecker, interpreter)  │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐    ┌──────────┐    ┌─────────┐
    │  PyO3   │    │  napi-rs │    │  WASM   │
    │ Binding │    │  Binding │    │ Binding │
    └────┬────┘    └────┬─────┘    └────┬────┘
         │              │               │
         ▼              ▼               ▼
    ┌─────────┐    ┌──────────┐    ┌─────────────┐
    │ Python  │    │  Node.js │    │  Browser /  │
    │ Package │    │  Package │    │  Any WASM   │
    └─────────┘    └──────────┘    └─────────────┘
```

### Build Configuration

```toml
# Cargo.toml (workspace)
[workspace]
members = [
    "hoist-core",
    "hoist-python",
    "hoist-node",
    "hoist-wasm",
    "hoist-cli",
]

# hoist-python/Cargo.toml
[package]
name = "hoist-python"
version = "0.1.0"
edition = "2021"

[lib]
name = "hoist"
crate-type = ["cdylib"]

[dependencies]
hoist-core = { path = "../hoist-core" }
pyo3 = { version = "0.20", features = ["extension-module"] }

# hoist-node/Cargo.toml
[package]
name = "hoist-node"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
hoist-core = { path = "../hoist-core" }
napi = "2"
napi-derive = "2"

[build-dependencies]
napi-build = "2"

# hoist-wasm/Cargo.toml
[package]
name = "hoist-wasm"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
hoist-core = { path = "../hoist-core" }
wasm-bindgen = "0.2"
js-sys = "0.3"
```

### CI/CD Pipeline

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags: ['v*']

jobs:
  build-python:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.9', '3.10', '3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      - uses: PyO3/maturin-action@v1
        with:
          command: build
          args: --release -m hoist-python/Cargo.toml
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}-${{ matrix.python-version }}
          path: target/wheels/

  build-node:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      - run: npm install
        working-directory: hoist-node
      - run: npm run build
        working-directory: hoist-node

  build-wasm:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          target: wasm32-unknown-unknown
      - run: cargo install wasm-pack
      - run: wasm-pack build --target web hoist-wasm

  publish-pypi:
    needs: build-python
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: wheels/

  publish-npm:
    needs: build-node
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-node@v4
        with:
          registry-url: 'https://registry.npmjs.org'
      - run: npm publish
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

---

## 11. Testing Strategy

### Test Suite Structure

```
tests/
├── compliance/           # Language compliance tests (JSON format)
│   ├── lexer/
│   │   ├── tokens.json
│   │   ├── strings.json
│   │   └── numbers.json
│   ├── parser/
│   │   ├── expressions.json
│   │   ├── statements.json
│   │   └── errors.json
│   ├── typechecker/
│   │   ├── inference.json
│   │   ├── errors.json
│   │   └── stdlib.json
│   └── interpreter/
│       ├── basic.json
│       ├── stdlib.json
│       ├── limits.json
│       └── errors.json
├── integration/          # Integration tests
│   ├── python/
│   ├── node/
│   └── wasm/
└── fuzz/                 # Fuzz testing
    ├── lexer_fuzz.rs
    ├── parser_fuzz.rs
    └── eval_fuzz.rs
```

### Compliance Test Format

```json
{
  "name": "basic_arithmetic",
  "version": "1.0",
  "tests": [
    {
      "name": "addition",
      "input": "return 1 + 2",
      "expected": {"type": "int", "value": 3}
    },
    {
      "name": "string_concat",
      "input": "return \"hello\" + \" \" + \"world\"",
      "expected": {"type": "string", "value": "hello world"}
    },
    {
      "name": "pipeline_map",
      "input": "return [1, 2, 3] |> map with (x -> x * 2)",
      "expected": {"type": "list", "value": [2, 4, 6]}
    },
    {
      "name": "division_by_zero",
      "input": "return 1 / 0",
      "expected_error": "DivisionByZero"
    }
  ]
}
```

### Test Runner

```rust
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct TestSuite {
    name: String,
    version: String,
    tests: Vec<TestCase>,
}

#[derive(Deserialize)]
struct TestCase {
    name: String,
    input: String,
    context: Option<HashMap<String, Value>>,
    expected: Option<ExpectedValue>,
    expected_error: Option<String>,
}

#[derive(Deserialize)]
struct ExpectedValue {
    #[serde(rename = "type")]
    ty: String,
    value: serde_json::Value,
}

fn run_compliance_tests(path: &str) {
    let content = fs::read_to_string(path).unwrap();
    let suite: TestSuite = serde_json::from_str(&content).unwrap();

    println!("Running suite: {} (v{})", suite.name, suite.version);

    let mut passed = 0;
    let mut failed = 0;

    for test in &suite.tests {
        let result = run_test(test);
        match result {
            Ok(()) => {
                passed += 1;
                println!("  ✓ {}", test.name);
            }
            Err(e) => {
                failed += 1;
                println!("  ✗ {}: {}", test.name, e);
            }
        }
    }

    println!("\nResults: {} passed, {} failed", passed, failed);
}

fn run_test(test: &TestCase) -> Result<(), String> {
    let host = Box::new(MockHost::new());
    let mut interp = Interpreter::new(host);

    // Set context
    if let Some(ctx) = &test.context {
        for (k, v) in ctx {
            interp.set_var(k, v.clone());
        }
    }

    // Parse and run
    let program = parse(&test.input)
        .map_err(|e| format!("Parse error: {:?}", e))?;

    let result = interp.eval(&program);

    // Check expectations
    if let Some(expected) = &test.expected {
        let val = result.map_err(|e| format!("Runtime error: {:?}", e))?;
        check_value(&val, expected)?;
    } else if let Some(expected_error) = &test.expected_error {
        match result {
            Err(e) => {
                if !format!("{:?}", e).contains(expected_error) {
                    return Err(format!("Expected error '{}', got '{:?}'", expected_error, e));
                }
            }
            Ok(v) => {
                return Err(format!("Expected error '{}', got value {:?}", expected_error, v));
            }
        }
    }

    Ok(())
}
```

### Property-Based Testing

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn lexer_never_panics(input in ".*") {
        let _ = lex(&input);
    }

    #[test]
    fn parser_never_panics(input in ".*") {
        let _ = parse(&input);
    }

    #[test]
    fn map_preserves_length(list in prop::collection::vec(any::<i64>(), 0..100)) {
        let input = format!("return {:?} |> map with (x -> x * 2)", list);
        let result = eval_to_list(&input).unwrap();
        prop_assert_eq!(result.len(), list.len());
    }

    #[test]
    fn filter_shrinks_or_preserves(list in prop::collection::vec(any::<i64>(), 0..100)) {
        let input = format!("return {:?} |> filter where it > 0", list);
        let result = eval_to_list(&input).unwrap();
        prop_assert!(result.len() <= list.len());
    }
}
```

---

## 12. Performance Considerations

### Optimization Opportunities

**Constant Folding**
```rust
fn fold_constants(expr: &Expr) -> Expr {
    match expr {
        Expr::BinOp { op, left, right } => {
            let left = fold_constants(left);
            let right = fold_constants(right);

            // If both are literals, compute at compile time
            if let (Expr::Literal(l), Expr::Literal(r)) = (&left, &right) {
                if let Some(result) = eval_const_binop(*op, l, r) {
                    return Expr::Literal(result);
                }
            }

            Expr::BinOp {
                op: *op,
                left: Box::new(left),
                right: Box::new(right),
            }
        }
        // ... other cases
    }
}
```

**Common Subexpression Elimination**
```rust
fn eliminate_cse(program: &mut IRProgram) {
    let mut seen: HashMap<IR, usize> = HashMap::new();
    // Track and reuse common subexpressions
}
```

**Tail Call Optimization**
```rust
fn is_tail_position(expr: &Expr, in_return: bool) -> bool {
    match expr {
        Expr::Call { .. } if in_return => true,
        Expr::If { then_branch, else_branch, .. } => {
            is_tail_position(then_branch, in_return) &&
            is_tail_position(else_branch, in_return)
        }
        _ => false,
    }
}
```

### Memory Management

- Use arena allocation for AST nodes
- Implement copy-on-write for large values
- Pool string allocations
- Consider reference counting for closures

### Benchmarking

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_map(c: &mut Criterion) {
    let runtime = create_runtime();
    let program = "[1,2,3,4,5,6,7,8,9,10] |> map with (x -> x * 2)";

    c.bench_function("map_10_elements", |b| {
        b.iter(|| runtime.eval(program))
    });
}

fn benchmark_nested_pipelines(c: &mut Criterion) {
    let runtime = create_runtime();
    let program = r#"
        range(1, 100)
        |> map with (x -> x * 2)
        |> filter where it > 50
        |> map with (x -> x + 1)
        |> reduce 0 with (a, b -> a + b)
    "#;

    c.bench_function("nested_pipeline_100", |b| {
        b.iter(|| runtime.eval(program))
    });
}

criterion_group!(benches, benchmark_map, benchmark_nested_pipelines);
criterion_main!(benches);
```

---

## 13. Reference Implementation Roadmap

### Phase 1: Core Language (Weeks 1-4)
- [ ] Lexer with string interpolation
- [ ] Parser (all expressions and statements)
- [ ] AST definition
- [ ] Basic type checker
- [ ] Tree-walking interpreter
- [ ] Core stdlib (strings, lists)

### Phase 2: Safety & Limits (Weeks 5-6)
- [ ] Recursion depth limits
- [ ] Iteration counting
- [ ] Timeout support
- [ ] Memory limits
- [ ] RE2 regex integration

### Phase 3: Host Integration (Weeks 7-8)
- [ ] Host callback interface
- [ ] `ask` function implementation
- [ ] Logging hooks
- [ ] Cancellation support

### Phase 4: Python Binding (Weeks 9-10)
- [ ] PyO3 wrapper
- [ ] Value conversion
- [ ] Async callback support
- [ ] Package build (maturin)
- [ ] Documentation

### Phase 5: Additional Bindings (Weeks 11-14)
- [ ] Node.js binding (napi-rs)
- [ ] WASM build
- [ ] CLI tool

### Phase 6: Polish (Weeks 15-16)
- [ ] Comprehensive test suite
- [ ] Performance optimization
- [ ] Error message improvements
- [ ] Documentation site
- [ ] Example projects

### Milestone Checklist

**v0.1.0 - Alpha**
- Core language working
- Basic interpreter
- Python binding functional

**v0.5.0 - Beta**
- All stdlib functions
- Full type inference
- Node.js binding
- Compliance test suite

**v1.0.0 - Stable**
- Production-ready safety guarantees
- Complete documentation
- All bindings stable
- Performance benchmarks

---

## Appendix: Quick Reference

### File Locations (Recommended)

| Component | Path |
|-----------|------|
| Grammar (pest) | `hoist-core/src/hoist.pest` |
| Lexer | `hoist-core/src/lexer.rs` |
| Parser | `hoist-core/src/parser.rs` |
| AST | `hoist-core/src/ast.rs` |
| Types | `hoist-core/src/types.rs` |
| Type Checker | `hoist-core/src/typecheck.rs` |
| Interpreter | `hoist-core/src/interpreter.rs` |
| Stdlib | `hoist-core/src/stdlib.rs` |
| Errors | `hoist-core/src/error.rs` |
| Python Binding | `hoist-python/src/lib.rs` |
| Node Binding | `hoist-node/src/lib.rs` |
| WASM Binding | `hoist-wasm/src/lib.rs` |

### Key Dependencies

```toml
# Rust
logos = "0.13"          # Lexer
pest = "2.7"            # Parser (alternative)
regex = "1.10"          # RE2-compatible regex
serde = "1.0"           # Serialization
serde_json = "1.0"      # JSON
thiserror = "1.0"       # Error handling
pyo3 = "0.20"           # Python binding
napi = "2"              # Node binding
wasm-bindgen = "0.2"    # WASM binding
```

### Command Cheatsheet

```bash
# Build all
cargo build --workspace --release

# Run tests
cargo test --workspace

# Build Python wheel
cd hoist-python && maturin build --release

# Build Node package
cd hoist-node && npm run build

# Build WASM
wasm-pack build --target web hoist-wasm

# Run benchmarks
cargo bench

# Run compliance tests
cargo run --bin compliance-runner -- tests/compliance/
```
