use crate::ast::*;
use crate::error::{HoistError, Span};
use crate::lexer::{Token, TokenKind, StringFragment};

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    pub fn parse_program(&mut self) -> Result<Program, HoistError> {
        let mut bindings = Vec::new();

        // Parse bindings until we hit `return`
        while !self.at_end() && !self.check(&TokenKind::Return) {
            bindings.push(self.parse_binding()?);
        }

        // Parse return expression
        self.expect(&TokenKind::Return)?;
        let return_expr = self.parse_expr()?;

        Ok(Program { bindings, return_expr })
    }

    // ---------------------------------------------------------------
    // Bindings
    // ---------------------------------------------------------------
    fn parse_binding(&mut self) -> Result<Binding, HoistError> {
        let span = self.current_span();
        self.expect(&TokenKind::Let)?;
        let name = self.expect_ident()?;

        // Optional type annotation
        let type_ann = if self.match_tok(&TokenKind::Colon) {
            Some(self.parse_type_expr()?)
        } else {
            None
        };

        self.expect(&TokenKind::Eq)?;
        let value = self.parse_expr()?;

        Ok(Binding { name, type_ann, value, span })
    }

    // ---------------------------------------------------------------
    // Expressions — precedence climbing
    // ---------------------------------------------------------------
    pub fn parse_expr(&mut self) -> Result<Expr, HoistError> {
        self.parse_pipe()
    }

    /// Pipe: lowest precedence  `expr |> expr`
    fn parse_pipe(&mut self) -> Result<Expr, HoistError> {
        let mut left = self.parse_or()?;

        while self.match_tok(&TokenKind::Pipe) {
            // After |>, parse a pipeline stage: keyword-form or function call
            let right = self.parse_pipe_stage(&left)?;
            left = right;
        }

        Ok(left)
    }

    /// Parse right side of a |> pipe, threading `left` as the first argument.
    fn parse_pipe_stage(&mut self, left: &Expr) -> Result<Expr, HoistError> {
        let span = self.current_span();

        // The pipe RHS may be a collection keyword (map, filter, split, etc.)
        // or a function call.
        if self.check_keyword_op() {
            // Parse the collection op, injecting `left` as the collection arg
            self.parse_collection_op_with(left.clone())
        } else {
            // Regular function call: expr |> f  =>  f(expr)
            let func = self.parse_primary()?;
            Ok(Expr::new(
                ExprKind::Call {
                    func: Box::new(func),
                    args: vec![left.clone()],
                },
                span,
            ))
        }
    }

    fn parse_or(&mut self) -> Result<Expr, HoistError> {
        let mut left = self.parse_and()?;
        while self.match_tok(&TokenKind::Or) {
            let right = self.parse_and()?;
            let span = left.span;
            left = Expr::new(ExprKind::BinOp { op: BinOp::Or, left: Box::new(left), right: Box::new(right) }, span);
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, HoistError> {
        let mut left = self.parse_not()?;
        while self.match_tok(&TokenKind::And) {
            let right = self.parse_not()?;
            let span = left.span;
            left = Expr::new(ExprKind::BinOp { op: BinOp::And, left: Box::new(left), right: Box::new(right) }, span);
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<Expr, HoistError> {
        if self.match_tok(&TokenKind::Not) {
            let span = self.prev_span();
            let operand = self.parse_not()?;
            Ok(Expr::new(ExprKind::UnaryOp { op: UnaryOp::Not, operand: Box::new(operand) }, span))
        } else {
            self.parse_comparison()
        }
    }

    fn parse_comparison(&mut self) -> Result<Expr, HoistError> {
        let left = self.parse_addition()?;

        let op = match self.peek_kind() {
            Some(TokenKind::EqEq) => Some(BinOp::Eq),
            Some(TokenKind::BangEq) => Some(BinOp::Ne),
            Some(TokenKind::Lt) => Some(BinOp::Lt),
            Some(TokenKind::Le) => Some(BinOp::Le),
            Some(TokenKind::Gt) => Some(BinOp::Gt),
            Some(TokenKind::Ge) => Some(BinOp::Ge),
            _ => None,
        };

        if let Some(op) = op {
            self.advance();
            let right = self.parse_addition()?;
            let span = left.span;
            Ok(Expr::new(ExprKind::BinOp { op, left: Box::new(left), right: Box::new(right) }, span))
        } else {
            Ok(left)
        }
    }

    fn parse_addition(&mut self) -> Result<Expr, HoistError> {
        let mut left = self.parse_multiplication()?;

        loop {
            let op = match self.peek_kind() {
                Some(TokenKind::Plus) => Some(BinOp::Add),
                Some(TokenKind::Minus) => Some(BinOp::Sub),
                Some(TokenKind::PlusPlus) => Some(BinOp::Concat),
                _ => None,
            };

            if let Some(op) = op {
                self.advance();
                let right = self.parse_multiplication()?;
                let span = left.span;
                left = Expr::new(ExprKind::BinOp { op, left: Box::new(left), right: Box::new(right) }, span);
            } else {
                break;
            }
        }

        Ok(left)
    }

    fn parse_multiplication(&mut self) -> Result<Expr, HoistError> {
        let mut left = self.parse_unary()?;

        loop {
            let op = match self.peek_kind() {
                Some(TokenKind::Star) => Some(BinOp::Mul),
                Some(TokenKind::Slash) => Some(BinOp::Div),
                Some(TokenKind::Percent) => Some(BinOp::Mod),
                _ => None,
            };

            if let Some(op) = op {
                self.advance();
                let right = self.parse_unary()?;
                let span = left.span;
                left = Expr::new(ExprKind::BinOp { op, left: Box::new(left), right: Box::new(right) }, span);
            } else {
                break;
            }
        }

        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, HoistError> {
        if self.match_tok(&TokenKind::Minus) {
            let span = self.prev_span();
            let operand = self.parse_unary()?;
            Ok(Expr::new(ExprKind::UnaryOp { op: UnaryOp::Neg, operand: Box::new(operand) }, span))
        } else {
            self.parse_postfix()
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr, HoistError> {
        let mut expr = self.parse_primary()?;

        loop {
            if self.match_tok(&TokenKind::LBracket) {
                // Index: expr[index]
                let index = self.parse_expr()?;
                self.expect(&TokenKind::RBracket)?;
                let span = expr.span;
                expr = Expr::new(ExprKind::Index { base: Box::new(expr), index: Box::new(index) }, span);
            } else if self.match_tok(&TokenKind::Dot) {
                // Member: expr.field
                let field = self.expect_ident()?;
                let span = expr.span;
                expr = Expr::new(ExprKind::Member { base: Box::new(expr), field }, span);
            } else if self.match_tok(&TokenKind::LParen) {
                // Call: expr(args...)
                let args = self.parse_args()?;
                self.expect(&TokenKind::RParen)?;
                let span = expr.span;
                expr = Expr::new(ExprKind::Call { func: Box::new(expr), args }, span);
            } else {
                break;
            }
        }

        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, HoistError> {
        let span = self.current_span();

        match self.peek_kind() {
            // Integer literal
            Some(TokenKind::Int(_)) => {
                if let TokenKind::Int(n) = self.advance_and_get().kind {
                    Ok(Expr::new(ExprKind::IntLit(n), span))
                } else {
                    unreachable!()
                }
            }

            // Boolean literals
            Some(TokenKind::True) => {
                self.advance();
                Ok(Expr::new(ExprKind::BoolLit(true), span))
            }
            Some(TokenKind::False) => {
                self.advance();
                Ok(Expr::new(ExprKind::BoolLit(false), span))
            }

            // None literal
            Some(TokenKind::None) => {
                self.advance();
                // Represent None as a constructor-like variant
                Ok(Expr::new(ExprKind::Var("None".into()), span))
            }

            // Some(expr) constructor
            Some(TokenKind::Some) => {
                self.advance();
                self.expect(&TokenKind::LParen)?;
                let inner = self.parse_expr()?;
                self.expect(&TokenKind::RParen)?;
                Ok(Expr::new(
                    ExprKind::Call {
                        func: Box::new(Expr::new(ExprKind::Var("Some".into()), span)),
                        args: vec![inner],
                    },
                    span,
                ))
            }

            // String literal
            Some(TokenKind::StringLit(_)) => {
                if let TokenKind::StringLit(fragments) = self.advance_and_get().kind {
                    let parts = self.convert_string_fragments(fragments)?;
                    Ok(Expr::new(ExprKind::StringLit(parts), span))
                } else {
                    unreachable!()
                }
            }

            // Identifier (variable or lambda)
            Some(TokenKind::Ident(_)) => {
                let name = self.expect_ident()?;

                // Check if this is a lambda: `name ->` or `name →`
                if self.check(&TokenKind::Arrow) {
                    self.advance();
                    let body = self.parse_expr()?;
                    Ok(Expr::new(ExprKind::Lambda { params: vec![name], body: Box::new(body) }, span))
                } else {
                    Ok(Expr::new(ExprKind::Var(name), span))
                }
            }

            // Parenthesized expression, tuple, or multi-param lambda
            Some(TokenKind::LParen) => {
                self.advance();
                if self.check(&TokenKind::RParen) {
                    self.advance();
                    // Unit / empty tuple
                    return Ok(Expr::new(ExprKind::Tuple(vec![]), span));
                }

                let first = self.parse_expr()?;

                if self.match_tok(&TokenKind::Comma) {
                    // Tuple
                    let mut items = vec![first];
                    items.push(self.parse_expr()?);
                    while self.match_tok(&TokenKind::Comma) {
                        items.push(self.parse_expr()?);
                    }
                    self.expect(&TokenKind::RParen)?;
                    Ok(Expr::new(ExprKind::Tuple(items), span))
                } else {
                    self.expect(&TokenKind::RParen)?;
                    Ok(first) // Parenthesized expression
                }
            }

            // List literal
            Some(TokenKind::LBracket) => {
                self.advance();
                let mut items = Vec::new();
                if !self.check(&TokenKind::RBracket) {
                    items.push(self.parse_expr()?);
                    while self.match_tok(&TokenKind::Comma) {
                        if self.check(&TokenKind::RBracket) { break; }
                        items.push(self.parse_expr()?);
                    }
                }
                self.expect(&TokenKind::RBracket)?;
                Ok(Expr::new(ExprKind::List(items), span))
            }

            // Record literal { field: value, ... }
            Some(TokenKind::LBrace) => {
                self.advance();
                let mut fields = Vec::new();
                if !self.check(&TokenKind::RBrace) {
                    let name = self.expect_ident()?;
                    self.expect(&TokenKind::Colon)?;
                    let value = self.parse_expr()?;
                    fields.push((name, value));
                    while self.match_tok(&TokenKind::Comma) {
                        if self.check(&TokenKind::RBrace) { break; }
                        let name = self.expect_ident()?;
                        self.expect(&TokenKind::Colon)?;
                        let value = self.parse_expr()?;
                        fields.push((name, value));
                    }
                }
                self.expect(&TokenKind::RBrace)?;
                Ok(Expr::new(ExprKind::Record(fields), span))
            }

            // `if` expression
            Some(TokenKind::If) => self.parse_if(),

            // `match` expression
            Some(TokenKind::Match) => self.parse_match(),

            // Collection operations as expressions (not in pipe)
            Some(TokenKind::Map) => self.parse_map_expr(),
            Some(TokenKind::Filter) => self.parse_filter_expr(),
            Some(TokenKind::Fold) => self.parse_fold_expr(),
            Some(TokenKind::Take) => self.parse_take_expr(),
            Some(TokenKind::Drop) => self.parse_drop_expr(),
            Some(TokenKind::Split) => self.parse_split_expr(),
            Some(TokenKind::Join) => self.parse_join_expr(),
            Some(TokenKind::Window) => self.parse_window_expr(),
            Some(TokenKind::Slice) => self.parse_slice_expr(),

            // `ask` expression
            Some(TokenKind::Ask) => self.parse_ask_expr(),

            Some(TokenKind::Eof) | None => {
                Err(HoistError::UnexpectedEof { span: Some(span) })
            }

            _ => {
                let tok = self.advance_and_get();
                Err(HoistError::ParseError {
                    expected: "expression".into(),
                    found: format!("{:?}", tok.kind),
                    span: tok.span,
                })
            }
        }
    }

    // ---------------------------------------------------------------
    // If / Match
    // ---------------------------------------------------------------
    fn parse_if(&mut self) -> Result<Expr, HoistError> {
        let span = self.current_span();
        self.expect(&TokenKind::If)?;
        let condition = self.parse_expr()?;
        self.expect(&TokenKind::Then)?;
        let then_branch = self.parse_expr()?;
        self.expect(&TokenKind::Else)?;
        let else_branch = self.parse_expr()?;

        Ok(Expr::new(
            ExprKind::If {
                condition: Box::new(condition),
                then_branch: Box::new(then_branch),
                else_branch: Box::new(else_branch),
            },
            span,
        ))
    }

    fn parse_match(&mut self) -> Result<Expr, HoistError> {
        let span = self.current_span();
        self.expect(&TokenKind::Match)?;
        let scrutinee = self.parse_expr()?;
        self.expect(&TokenKind::With)?;

        let mut arms = Vec::new();
        while self.match_tok(&TokenKind::Bar) {
            let arm_span = self.current_span();
            let pattern = self.parse_pattern()?;
            self.expect(&TokenKind::Arrow)?;
            let body = self.parse_expr()?;
            arms.push(MatchArm { pattern, body, span: arm_span });
        }

        if arms.is_empty() {
            return Err(HoistError::ParseError {
                expected: "at least one match arm".into(),
                found: format!("{:?}", self.peek_kind()),
                span,
            });
        }

        Ok(Expr::new(ExprKind::Match { scrutinee: Box::new(scrutinee), arms }, span))
    }

    fn parse_pattern(&mut self) -> Result<Pattern, HoistError> {
        let span = self.current_span();

        // Wildcard
        if self.check_ident("_") {
            self.advance();
            return Ok(Pattern::Wildcard);
        }

        // Boolean patterns
        if self.match_tok(&TokenKind::True) {
            return Ok(Pattern::BoolLit(true));
        }
        if self.match_tok(&TokenKind::False) {
            return Ok(Pattern::BoolLit(false));
        }

        // None
        if self.match_tok(&TokenKind::None) {
            return Ok(Pattern::Constructor { name: "None".into(), args: vec![] });
        }

        // Some(pattern)
        if self.match_tok(&TokenKind::Some) {
            self.expect(&TokenKind::LParen)?;
            let inner = self.parse_pattern()?;
            self.expect(&TokenKind::RParen)?;
            return Ok(Pattern::Constructor { name: "Some".into(), args: vec![inner] });
        }

        // Integer
        if let Some(TokenKind::Int(_)) = self.peek_kind() {
            if let TokenKind::Int(n) = self.advance_and_get().kind {
                return Ok(Pattern::IntLit(n));
            }
        }

        // String
        if let Some(TokenKind::StringLit(_)) = self.peek_kind() {
            if let TokenKind::StringLit(frags) = self.advance_and_get().kind {
                // Only support non-interpolated strings in patterns
                if frags.len() == 1 {
                    if let StringFragment::Literal(s) = &frags[0] {
                        return Ok(Pattern::StringLit(s.clone()));
                    }
                }
                return Err(HoistError::ParseError {
                    expected: "simple string (no interpolation) in pattern".into(),
                    found: "interpolated string".into(),
                    span,
                });
            }
        }

        // Identifier (variable binding or constructor)
        if let Some(TokenKind::Ident(_)) = self.peek_kind() {
            let name = self.expect_ident()?;
            if self.match_tok(&TokenKind::LParen) {
                // Constructor pattern
                let mut args = Vec::new();
                if !self.check(&TokenKind::RParen) {
                    args.push(self.parse_pattern()?);
                    while self.match_tok(&TokenKind::Comma) {
                        args.push(self.parse_pattern()?);
                    }
                }
                self.expect(&TokenKind::RParen)?;
                Ok(Pattern::Constructor { name, args })
            } else {
                Ok(Pattern::Var(name))
            }
        } else {
            Err(HoistError::ParseError {
                expected: "pattern".into(),
                found: format!("{:?}", self.peek_kind()),
                span,
            })
        }
    }

    // ---------------------------------------------------------------
    // Collection operations
    // ---------------------------------------------------------------
    fn check_keyword_op(&self) -> bool {
        matches!(
            self.peek_kind(),
            Some(TokenKind::Map | TokenKind::Filter | TokenKind::Fold |
                 TokenKind::Take | TokenKind::Drop | TokenKind::Split |
                 TokenKind::Join | TokenKind::Window | TokenKind::Slice)
        )
    }

    /// Parse a collection operation where the collection is provided by the pipe.
    fn parse_collection_op_with(&mut self, collection: Expr) -> Result<Expr, HoistError> {
        let span = self.current_span();
        match self.peek_kind() {
            Some(TokenKind::Map) => {
                self.advance();
                self.expect(&TokenKind::With)?;
                let transform = self.parse_lambda_or_expr()?;
                Ok(Expr::new(ExprKind::Map { collection: Box::new(collection), transform: Box::new(transform) }, span))
            }
            Some(TokenKind::Filter) => {
                self.advance();
                self.expect(&TokenKind::Where)?;
                let predicate = self.parse_lambda_or_expr()?;
                Ok(Expr::new(ExprKind::Filter { collection: Box::new(collection), predicate: Box::new(predicate) }, span))
            }
            Some(TokenKind::Split) => {
                self.advance();
                self.expect(&TokenKind::By)?;
                let delim = self.parse_or()?;
                Ok(Expr::new(ExprKind::Split { text: Box::new(collection), delimiter: Box::new(delim) }, span))
            }
            Some(TokenKind::Join) => {
                self.advance();
                self.expect(&TokenKind::With)?;
                let sep = self.parse_or()?;
                Ok(Expr::new(ExprKind::Join { list: Box::new(collection), separator: Box::new(sep) }, span))
            }
            Some(TokenKind::Fold) => {
                self.advance();
                self.expect(&TokenKind::From)?;
                let initial = self.parse_postfix()?;
                self.expect(&TokenKind::With)?;
                let acc_name = self.expect_ident()?;
                self.expect(&TokenKind::Comma)?;
                let elem_name = self.expect_ident()?;
                self.expect(&TokenKind::Arrow)?;
                let body = self.parse_or()?; // don't consume |>
                Ok(Expr::new(ExprKind::Fold {
                    collection: Box::new(collection),
                    initial: Box::new(initial),
                    acc_name,
                    elem_name,
                    body: Box::new(body),
                }, span))
            }
            Some(TokenKind::Take) => {
                self.advance();
                let count = self.parse_postfix()?;
                Ok(Expr::new(ExprKind::Take { count: Box::new(count), collection: Box::new(collection) }, span))
            }
            Some(TokenKind::Drop) => {
                self.advance();
                let count = self.parse_postfix()?;
                Ok(Expr::new(ExprKind::Drop { count: Box::new(count), collection: Box::new(collection) }, span))
            }
            Some(TokenKind::Window) => {
                self.advance();
                self.expect(&TokenKind::Size)?;
                let size = self.parse_postfix()?;
                self.expect(&TokenKind::Stride)?;
                let stride = self.parse_postfix()?;
                Ok(Expr::new(ExprKind::Window { text: Box::new(collection), size: Box::new(size), stride: Box::new(stride) }, span))
            }
            Some(TokenKind::Slice) => {
                self.advance();
                self.expect(&TokenKind::From)?;
                let start = self.parse_unary()?;
                self.expect(&TokenKind::To)?;
                let end = self.parse_unary()?;
                Ok(Expr::new(ExprKind::Slice { text: Box::new(collection), start: Box::new(start), end: Box::new(end) }, span))
            }
            _ => {
                let tok = self.advance_and_get();
                Err(HoistError::ParseError {
                    expected: "collection operation after |>".into(),
                    found: format!("{:?}", tok.kind),
                    span: tok.span,
                })
            }
        }
    }

    /// `map collection with transform`
    fn parse_map_expr(&mut self) -> Result<Expr, HoistError> {
        let span = self.current_span();
        self.expect(&TokenKind::Map)?;
        let collection = self.parse_postfix()?;
        self.expect(&TokenKind::With)?;
        let transform = self.parse_lambda_or_expr()?;
        Ok(Expr::new(ExprKind::Map { collection: Box::new(collection), transform: Box::new(transform) }, span))
    }

    /// `filter collection where predicate`
    fn parse_filter_expr(&mut self) -> Result<Expr, HoistError> {
        let span = self.current_span();
        self.expect(&TokenKind::Filter)?;
        let collection = self.parse_postfix()?;
        self.expect(&TokenKind::Where)?;
        let predicate = self.parse_lambda_or_expr()?;
        Ok(Expr::new(ExprKind::Filter { collection: Box::new(collection), predicate: Box::new(predicate) }, span))
    }

    /// `fold collection from initial with acc, elem -> body`
    fn parse_fold_expr(&mut self) -> Result<Expr, HoistError> {
        let span = self.current_span();
        self.expect(&TokenKind::Fold)?;
        let collection = self.parse_postfix()?;
        self.expect(&TokenKind::From)?;
        let initial = self.parse_postfix()?;
        self.expect(&TokenKind::With)?;
        let acc_name = self.expect_ident()?;
        self.expect(&TokenKind::Comma)?;
        let elem_name = self.expect_ident()?;
        self.expect(&TokenKind::Arrow)?;
        let body = self.parse_expr()?;
        Ok(Expr::new(ExprKind::Fold {
            collection: Box::new(collection),
            initial: Box::new(initial),
            acc_name,
            elem_name,
            body: Box::new(body),
        }, span))
    }

    /// `take n from collection`
    fn parse_take_expr(&mut self) -> Result<Expr, HoistError> {
        let span = self.current_span();
        self.expect(&TokenKind::Take)?;
        let count = self.parse_postfix()?;
        self.expect(&TokenKind::From)?;
        let collection = self.parse_postfix()?;
        Ok(Expr::new(ExprKind::Take { count: Box::new(count), collection: Box::new(collection) }, span))
    }

    /// `drop n from collection`
    fn parse_drop_expr(&mut self) -> Result<Expr, HoistError> {
        let span = self.current_span();
        self.expect(&TokenKind::Drop)?;
        let count = self.parse_postfix()?;
        self.expect(&TokenKind::From)?;
        let collection = self.parse_postfix()?;
        Ok(Expr::new(ExprKind::Drop { count: Box::new(count), collection: Box::new(collection) }, span))
    }

    /// `split text by delimiter`
    fn parse_split_expr(&mut self) -> Result<Expr, HoistError> {
        let span = self.current_span();
        self.expect(&TokenKind::Split)?;
        let text = self.parse_postfix()?;
        self.expect(&TokenKind::By)?;
        let delimiter = self.parse_postfix()?;
        Ok(Expr::new(ExprKind::Split { text: Box::new(text), delimiter: Box::new(delimiter) }, span))
    }

    /// `join list with separator`
    fn parse_join_expr(&mut self) -> Result<Expr, HoistError> {
        let span = self.current_span();
        self.expect(&TokenKind::Join)?;
        let list = self.parse_postfix()?;
        self.expect(&TokenKind::With)?;
        let separator = self.parse_postfix()?;
        Ok(Expr::new(ExprKind::Join { list: Box::new(list), separator: Box::new(separator) }, span))
    }

    /// `window text size n stride m`
    fn parse_window_expr(&mut self) -> Result<Expr, HoistError> {
        let span = self.current_span();
        self.expect(&TokenKind::Window)?;
        let text = self.parse_postfix()?;
        self.expect(&TokenKind::Size)?;
        let size = self.parse_postfix()?;
        self.expect(&TokenKind::Stride)?;
        let stride = self.parse_postfix()?;
        Ok(Expr::new(ExprKind::Window { text: Box::new(text), size: Box::new(size), stride: Box::new(stride) }, span))
    }

    /// `slice text from start to end`
    fn parse_slice_expr(&mut self) -> Result<Expr, HoistError> {
        let span = self.current_span();
        self.expect(&TokenKind::Slice)?;
        let text = self.parse_postfix()?;
        self.expect(&TokenKind::From)?;
        let start = self.parse_unary()?; // allow negative: -1
        self.expect(&TokenKind::To)?;
        let end = self.parse_unary()?; // allow negative: -1
        Ok(Expr::new(ExprKind::Slice { text: Box::new(text), start: Box::new(start), end: Box::new(end) }, span))
    }

    // ---------------------------------------------------------------
    // Ask expression
    // ---------------------------------------------------------------
    fn parse_ask_expr(&mut self) -> Result<Expr, HoistError> {
        let span = self.current_span();
        self.expect(&TokenKind::Ask)?;
        let prompt = self.parse_postfix()?;

        let mut modifiers = AskModifiers::default();

        // Parse optional modifiers in any order
        loop {
            if self.match_tok(&TokenKind::As) {
                modifiers.typed_output = Some(self.parse_type_expr()?);
            } else if self.match_tok(&TokenKind::Via) {
                let channel = self.expect_ident()?;
                modifiers.channel = Some(channel);
            } else if self.check(&TokenKind::With) && self.peek_at_kind(1) == Some(&TokenKind::Retries) {
                self.advance(); // consume 'with'
                self.advance(); // consume 'retries'
                self.expect(&TokenKind::Colon)?;
                if let Some(TokenKind::Int(_)) = self.peek_kind() {
                    if let TokenKind::Int(n) = self.advance_and_get().kind {
                        modifiers.retries = Some(n as u32);
                    }
                } else {
                    return Err(HoistError::ParseError {
                        expected: "integer for retries count".into(),
                        found: format!("{:?}", self.peek_kind()),
                        span: self.current_span(),
                    });
                }
            } else if self.match_tok(&TokenKind::Fallback) {
                let fallback = self.parse_expr()?;
                modifiers.fallback = Some(Box::new(fallback));
            } else {
                break;
            }
        }

        Ok(Expr::new(ExprKind::Ask { prompt: Box::new(prompt), modifiers }, span))
    }

    // ---------------------------------------------------------------
    // Type expressions (for annotations and `ask ... as Type`)
    // ---------------------------------------------------------------
    fn parse_type_expr(&mut self) -> Result<TypeExpr, HoistError> {
        let span = self.current_span();

        // Record type: { field: Type, ... }
        if self.match_tok(&TokenKind::LBrace) {
            let mut fields = Vec::new();
            if !self.check(&TokenKind::RBrace) {
                let name = self.expect_ident()?;
                self.expect(&TokenKind::Colon)?;
                let ty = self.parse_type_expr()?;
                fields.push((name, ty));
                while self.match_tok(&TokenKind::Comma) {
                    if self.check(&TokenKind::RBrace) { break; }
                    let name = self.expect_ident()?;
                    self.expect(&TokenKind::Colon)?;
                    let ty = self.parse_type_expr()?;
                    fields.push((name, ty));
                }
            }
            self.expect(&TokenKind::RBrace)?;
            return Ok(TypeExpr::Record(fields));
        }

        // Named type, possibly generic: String, List<T>, Optional<T>
        let name = self.expect_ident()?;

        if self.match_tok(&TokenKind::Lt) {
            // Generic type
            let inner = self.parse_type_expr()?;
            self.expect(&TokenKind::Gt)?;
            match name.as_str() {
                "List" => Ok(TypeExpr::List(Box::new(inner))),
                "Optional" => Ok(TypeExpr::Optional(Box::new(inner))),
                _ => Err(HoistError::ParseError {
                    expected: "List or Optional".into(),
                    found: name,
                    span,
                }),
            }
        } else {
            Ok(TypeExpr::Named(name))
        }
    }

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------

    /// Parse either an explicit lambda `x -> expr` or a bare expression.
    /// The result is always wrapped as a lambda for use in map/filter/etc.
    /// Uses `parse_or` (not `parse_pipe`) so that `|>` is not consumed.
    fn parse_lambda_or_expr(&mut self) -> Result<Expr, HoistError> {
        let span = self.current_span();

        // Check for explicit lambda: ident ->
        if let Some(TokenKind::Ident(_)) = self.peek_kind() {
            if self.peek_at_kind(1) == Some(&TokenKind::Arrow) {
                let param = self.expect_ident()?;
                self.advance(); // consume ->
                let body = self.parse_or()?; // don't consume |>
                return Ok(Expr::new(ExprKind::Lambda { params: vec![param], body: Box::new(body) }, span));
            }
        }

        // Parse expression at `or` precedence (stops before |>)
        let expr = self.parse_or()?;

        // Always wrap in an `it` lambda for use in map/filter/fold contexts.
        // If the body references `it`, that binding is used; otherwise `it` is simply unused.
        Ok(Expr::new(
            ExprKind::Lambda { params: vec!["it".into()], body: Box::new(expr) },
            span,
        ))
    }

    fn convert_string_fragments(&self, fragments: Vec<StringFragment>) -> Result<Vec<StringPart>, HoistError> {
        let mut parts = Vec::new();
        for frag in fragments {
            match frag {
                StringFragment::Literal(s) => parts.push(StringPart::Literal(s)),
                StringFragment::Interpolation(tokens) => {
                    let mut sub_parser = Parser::new(tokens);
                    let expr = sub_parser.parse_expr()?;
                    parts.push(StringPart::Interpolation(Box::new(expr)));
                }
            }
        }
        Ok(parts)
    }

    fn parse_args(&mut self) -> Result<Vec<Expr>, HoistError> {
        let mut args = Vec::new();
        if !self.check(&TokenKind::RParen) {
            args.push(self.parse_expr()?);
            while self.match_tok(&TokenKind::Comma) {
                args.push(self.parse_expr()?);
            }
        }
        Ok(args)
    }

    // --- Token manipulation ---

    fn peek_kind(&self) -> Option<TokenKind> {
        self.tokens.get(self.pos).map(|t| t.kind.clone())
    }

    fn peek_at_kind(&self, offset: usize) -> Option<&TokenKind> {
        self.tokens.get(self.pos + offset).map(|t| &t.kind)
    }

    fn check(&self, kind: &TokenKind) -> bool {
        self.peek_kind().as_ref() == Some(kind)
    }

    fn check_ident(&self, name: &str) -> bool {
        matches!(self.peek_kind(), Some(TokenKind::Ident(ref n)) if n == name)
    }

    fn match_tok(&mut self, kind: &TokenKind) -> bool {
        if self.check(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: &TokenKind) -> Result<(), HoistError> {
        if self.check(kind) {
            self.advance();
            Ok(())
        } else {
            let span = self.current_span();
            let found = self.peek_kind()
                .map(|k| format!("{:?}", k))
                .unwrap_or_else(|| "EOF".into());
            Err(HoistError::ParseError {
                expected: format!("{:?}", kind),
                found,
                span,
            })
        }
    }

    fn expect_ident(&mut self) -> Result<String, HoistError> {
        match self.peek_kind() {
            Some(TokenKind::Ident(name)) => {
                self.advance();
                Ok(name)
            }
            other => {
                Err(HoistError::ParseError {
                    expected: "identifier".into(),
                    found: format!("{:?}", other),
                    span: self.current_span(),
                })
            }
        }
    }

    fn advance(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn advance_and_get(&mut self) -> Token {
        let tok = self.tokens.get(self.pos).cloned().unwrap_or(Token {
            kind: TokenKind::Eof,
            span: Span::dummy(),
        });
        self.advance();
        tok
    }

    fn at_end(&self) -> bool {
        self.pos >= self.tokens.len() || matches!(self.peek_kind(), Some(TokenKind::Eof))
    }

    fn current_span(&self) -> Span {
        self.tokens.get(self.pos).map(|t| t.span).unwrap_or(Span::dummy())
    }

    fn prev_span(&self) -> Span {
        if self.pos > 0 {
            self.tokens[self.pos - 1].span
        } else {
            Span::dummy()
        }
    }
}

/// Convenience function: lex + parse.
pub fn parse_source(source: &str) -> Result<Program, HoistError> {
    let mut lexer = crate::lexer::Lexer::new(source);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> Program {
        parse_source(src).expect("parse failed")
    }

    #[test]
    fn test_simple_return() {
        let prog = parse("return 42");
        assert!(prog.bindings.is_empty());
        assert!(matches!(prog.return_expr.kind, ExprKind::IntLit(42)));
    }

    #[test]
    fn test_let_binding() {
        let prog = parse("let x = 42\nreturn x");
        assert_eq!(prog.bindings.len(), 1);
        assert_eq!(prog.bindings[0].name, "x");
    }

    #[test]
    fn test_arithmetic() {
        let prog = parse("return 1 + 2 * 3");
        // Should parse as 1 + (2 * 3) due to precedence
        match &prog.return_expr.kind {
            ExprKind::BinOp { op: BinOp::Add, right, .. } => {
                assert!(matches!(right.kind, ExprKind::BinOp { op: BinOp::Mul, .. }));
            }
            _ => panic!("expected addition"),
        }
    }

    #[test]
    fn test_string_interpolation() {
        let prog = parse(r#"let name = "world"
return "hello {name}!""#);
        assert_eq!(prog.bindings.len(), 1);
        match &prog.return_expr.kind {
            ExprKind::StringLit(parts) => {
                assert_eq!(parts.len(), 3);
            }
            _ => panic!("expected string literal"),
        }
    }

    #[test]
    fn test_if_expr() {
        let prog = parse("return if true then 1 else 2");
        assert!(matches!(prog.return_expr.kind, ExprKind::If { .. }));
    }

    #[test]
    fn test_list_literal() {
        let prog = parse("return [1, 2, 3]");
        match &prog.return_expr.kind {
            ExprKind::List(items) => assert_eq!(items.len(), 3),
            _ => panic!("expected list"),
        }
    }

    #[test]
    fn test_map_expression() {
        let prog = parse(r#"let items = [1, 2, 3]
return map items with it + 1"#);
        assert!(matches!(prog.return_expr.kind, ExprKind::Map { .. }));
    }

    #[test]
    fn test_filter_expression() {
        let prog = parse(r#"let items = [1, 2, 3]
return filter items where it > 1"#);
        assert!(matches!(prog.return_expr.kind, ExprKind::Filter { .. }));
    }

    #[test]
    fn test_split_by() {
        let prog = parse(r#"return split context by "\n""#);
        assert!(matches!(prog.return_expr.kind, ExprKind::Split { .. }));
    }

    #[test]
    fn test_ask_basic() {
        let prog = parse(r#"return ask "What is 2+2?""#);
        assert!(matches!(prog.return_expr.kind, ExprKind::Ask { .. }));
    }

    #[test]
    fn test_ask_with_modifiers() {
        let prog = parse(r#"return ask "List fruits" as List<String> via extractor with retries: 3 fallback []"#);
        match &prog.return_expr.kind {
            ExprKind::Ask { modifiers, .. } => {
                assert!(modifiers.typed_output.is_some());
                assert_eq!(modifiers.channel, Some("extractor".into()));
                assert_eq!(modifiers.retries, Some(3));
                assert!(modifiers.fallback.is_some());
            }
            _ => panic!("expected ask expression"),
        }
    }

    #[test]
    fn test_pipeline() {
        let prog = parse(r#"let items = [1, 2, 3]
return items |> filter where it > 1 |> map with it * 2"#);
        // The pipeline desugars into nested Map(Filter(...))
        assert!(matches!(prog.return_expr.kind, ExprKind::Map { .. }));
    }

    #[test]
    fn test_match_expression() {
        let prog = parse(r#"let x = Some(42)
return match x with
| Some(v) -> v
| None -> 0"#);
        assert!(matches!(prog.return_expr.kind, ExprKind::Match { .. }));
    }

    #[test]
    fn test_function_call() {
        let prog = parse("return length(items)");
        assert!(matches!(prog.return_expr.kind, ExprKind::Call { .. }));
    }

    #[test]
    fn test_fold_expression() {
        let prog = parse("let nums = [1, 2, 3]\nreturn fold nums from 0 with acc, n -> acc + n");
        assert!(matches!(prog.return_expr.kind, ExprKind::Fold { .. }));
    }

    #[test]
    fn test_record_literal() {
        let prog = parse(r#"return {name: "Alice", age: 30}"#);
        match &prog.return_expr.kind {
            ExprKind::Record(fields) => {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].0, "name");
                assert_eq!(fields[1].0, "age");
            }
            _ => panic!("expected record"),
        }
    }

    #[test]
    fn test_nested_let() {
        let prog = parse(r#"let a = 1
let b = 2
let c = a + b
return c"#);
        assert_eq!(prog.bindings.len(), 3);
    }
}
