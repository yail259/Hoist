use crate::error::{HoistError, Span};

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Keywords
    Let,
    In,
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
    As,
    Via,
    Retries,
    Fallback,
    Map,
    Filter,
    Fold,
    Take,
    Drop,
    Split,
    Join,
    Window,
    Slice,
    Ask,
    From,
    To,
    Size,
    Stride,
    Some,
    None,

    // Identifiers and literals
    Ident(String),
    Int(i64),
    /// A string consisting of parts: text segments and interpolation markers.
    /// The lexer emits a single StringLit token containing pre-split parts.
    StringLit(Vec<StringFragment>),

    // Operators
    Plus,       // +
    Minus,      // -
    Star,       // *
    Slash,      // /
    Percent,    // %
    EqEq,       // ==
    BangEq,     // !=
    Lt,         // <
    Le,         // <=
    Gt,         // >
    Ge,         // >=
    Pipe,       // |>
    PlusPlus,   // ++
    Eq,         // =
    Arrow,      // → or ->
    Bar,        // |
    Dot,        // .
    Colon,      // :
    Comma,      // ,

    // Delimiters
    LParen,     // (
    RParen,     // )
    LBracket,   // [
    RBracket,   // ]
    LBrace,     // {
    RBrace,     // }

    // Special
    Eof,
}

/// Fragment of a string literal (for interpolation support).
#[derive(Debug, Clone, PartialEq)]
pub enum StringFragment {
    Literal(String),
    Interpolation(Vec<Token>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }
}

pub struct Lexer {
    source: Vec<char>,
    pos: usize,
    line: usize,
    column: usize,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Self {
            source: source.chars().collect(),
            pos: 0,
            line: 1,
            column: 1,
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>, HoistError> {
        let mut tokens = Vec::new();
        loop {
            self.skip_whitespace_and_comments();
            if self.at_end() {
                tokens.push(Token::new(TokenKind::Eof, self.current_span()));
                break;
            }
            tokens.push(self.next_token()?);
        }
        Ok(tokens)
    }

    fn next_token(&mut self) -> Result<Token, HoistError> {
        let start = self.pos;
        let start_line = self.line;
        let start_col = self.column;

        let ch = self.peek().unwrap();

        let kind = match ch {
            // String literals
            '"' => return self.lex_string(),

            // Numbers
            '0'..='9' => return self.lex_number(),

            // Identifiers and keywords
            'a'..='z' | 'A'..='Z' | '_' => return self.lex_ident_or_keyword(),

            // Two-character operators (check first)
            '|' => {
                self.advance();
                if self.peek() == Some('>') {
                    self.advance();
                    TokenKind::Pipe
                } else {
                    TokenKind::Bar
                }
            }
            '+' => {
                self.advance();
                if self.peek() == Some('+') {
                    self.advance();
                    TokenKind::PlusPlus
                } else {
                    TokenKind::Plus
                }
            }
            '-' => {
                self.advance();
                if self.peek() == Some('>') {
                    self.advance();
                    TokenKind::Arrow
                } else if self.peek() == Some('-') {
                    // Line comment -- skip to end of line
                    self.skip_line_comment();
                    return self.next_token();
                } else {
                    TokenKind::Minus
                }
            }
            '=' => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::EqEq
                } else {
                    TokenKind::Eq
                }
            }
            '!' => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::BangEq
                } else {
                    return Err(HoistError::SyntaxError {
                        message: "unexpected '!', did you mean '!='?".into(),
                        span: self.span_from(start, start_line, start_col),
                    });
                }
            }
            '<' => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::Le
                } else {
                    TokenKind::Lt
                }
            }
            '>' => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::Ge
                } else {
                    TokenKind::Gt
                }
            }
            // Unicode arrow
            '\u{2192}' => {
                self.advance();
                TokenKind::Arrow
            }

            // Single-character tokens
            '*' => { self.advance(); TokenKind::Star }
            '/' => { self.advance(); TokenKind::Slash }
            '%' => { self.advance(); TokenKind::Percent }
            '.' => { self.advance(); TokenKind::Dot }
            ':' => { self.advance(); TokenKind::Colon }
            ',' => { self.advance(); TokenKind::Comma }
            '(' => { self.advance(); TokenKind::LParen }
            ')' => { self.advance(); TokenKind::RParen }
            '[' => { self.advance(); TokenKind::LBracket }
            ']' => { self.advance(); TokenKind::RBracket }
            '{' => {
                self.advance();
                if self.peek() == Some('-') {
                    // Block comment {- ... -}
                    self.advance(); // consume '-'
                    self.skip_block_comment()?;
                    return self.next_token();
                }
                TokenKind::LBrace
            }
            '}' => { self.advance(); TokenKind::RBrace }

            _ => {
                return Err(HoistError::SyntaxError {
                    message: format!("unexpected character '{ch}'"),
                    span: self.span_from(start, start_line, start_col),
                });
            }
        };

        Ok(Token::new(kind, self.span_from(start, start_line, start_col)))
    }

    fn lex_number(&mut self) -> Result<Token, HoistError> {
        let start = self.pos;
        let start_line = self.line;
        let start_col = self.column;

        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }

        let text: String = self.source[start..self.pos].iter().collect();
        let value: i64 = text.parse().map_err(|_| HoistError::SyntaxError {
            message: format!("invalid integer literal '{text}'"),
            span: self.span_from(start, start_line, start_col),
        })?;

        Ok(Token::new(
            TokenKind::Int(value),
            self.span_from(start, start_line, start_col),
        ))
    }

    fn lex_ident_or_keyword(&mut self) -> Result<Token, HoistError> {
        let start = self.pos;
        let start_line = self.line;
        let start_col = self.column;

        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        let text: String = self.source[start..self.pos].iter().collect();
        let kind = match text.as_str() {
            "let" => TokenKind::Let,
            "in" => TokenKind::In,
            "return" => TokenKind::Return,
            "if" => TokenKind::If,
            "then" => TokenKind::Then,
            "else" => TokenKind::Else,
            "match" => TokenKind::Match,
            "with" => TokenKind::With,
            "where" => TokenKind::Where,
            "by" => TokenKind::By,
            "and" => TokenKind::And,
            "or" => TokenKind::Or,
            "not" => TokenKind::Not,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "as" => TokenKind::As,
            "via" => TokenKind::Via,
            "retries" => TokenKind::Retries,
            "fallback" => TokenKind::Fallback,
            "map" => TokenKind::Map,
            "filter" => TokenKind::Filter,
            "fold" => TokenKind::Fold,
            "take" => TokenKind::Take,
            "drop" => TokenKind::Drop,
            "split" => TokenKind::Split,
            "join" => TokenKind::Join,
            "window" => TokenKind::Window,
            "slice" => TokenKind::Slice,
            "ask" => TokenKind::Ask,
            "from" => TokenKind::From,
            "to" => TokenKind::To,
            "size" => TokenKind::Size,
            "stride" => TokenKind::Stride,
            "Some" => TokenKind::Some,
            "None" => TokenKind::None,
            _ => TokenKind::Ident(text),
        };

        Ok(Token::new(kind, self.span_from(start, start_line, start_col)))
    }

    fn lex_string(&mut self) -> Result<Token, HoistError> {
        let start = self.pos;
        let start_line = self.line;
        let start_col = self.column;

        self.advance(); // consume opening "

        // Check for triple-quoted raw string
        if self.peek() == Some('"') && self.peek_at(1) == Some('"') {
            self.advance(); // second "
            self.advance(); // third "
            return self.lex_raw_string(start, start_line, start_col);
        }

        let mut fragments = Vec::new();
        let mut current_text = String::new();

        loop {
            match self.peek() {
                None => {
                    return Err(HoistError::UnterminatedString {
                        span: self.span_from(start, start_line, start_col),
                    });
                }
                Some('"') => {
                    self.advance();
                    if !current_text.is_empty() {
                        fragments.push(StringFragment::Literal(current_text));
                    }
                    break;
                }
                Some('\\') => {
                    self.advance();
                    let escaped = self.lex_escape_char(start, start_line, start_col)?;
                    current_text.push(escaped);
                }
                Some('{') => {
                    self.advance();
                    if !current_text.is_empty() {
                        fragments.push(StringFragment::Literal(current_text.clone()));
                        current_text.clear();
                    }
                    let interp_tokens = self.lex_interpolation()?;
                    fragments.push(StringFragment::Interpolation(interp_tokens));
                }
                Some(c) => {
                    if c == '\n' {
                        self.line += 1;
                        self.column = 0;
                    }
                    current_text.push(c);
                    self.advance();
                }
            }
        }

        Ok(Token::new(
            TokenKind::StringLit(fragments),
            self.span_from(start, start_line, start_col),
        ))
    }

    fn lex_raw_string(
        &mut self,
        start: usize,
        start_line: usize,
        start_col: usize,
    ) -> Result<Token, HoistError> {
        // We're past the opening """
        // Skip optional leading newline
        if self.peek() == Some('\n') {
            self.advance();
        }

        let mut text = String::new();

        loop {
            match self.peek() {
                None => {
                    return Err(HoistError::UnterminatedString {
                        span: self.span_from(start, start_line, start_col),
                    });
                }
                Some('"') if self.peek_at(1) == Some('"') && self.peek_at(2) == Some('"') => {
                    self.advance(); // "
                    self.advance(); // "
                    self.advance(); // "
                    // Trim trailing newline if present
                    if text.ends_with('\n') {
                        text.pop();
                    }
                    break;
                }
                Some(c) => {
                    if c == '\n' {
                        self.line += 1;
                        self.column = 0;
                    }
                    text.push(c);
                    self.advance();
                }
            }
        }

        Ok(Token::new(
            TokenKind::StringLit(vec![StringFragment::Literal(text)]),
            self.span_from(start, start_line, start_col),
        ))
    }

    fn lex_escape_char(
        &mut self,
        start: usize,
        start_line: usize,
        start_col: usize,
    ) -> Result<char, HoistError> {
        match self.peek() {
            Some('n') => { self.advance(); Ok('\n') }
            Some('r') => { self.advance(); Ok('\r') }
            Some('t') => { self.advance(); Ok('\t') }
            Some('\\') => { self.advance(); Ok('\\') }
            Some('"') => { self.advance(); Ok('"') }
            Some('{') => { self.advance(); Ok('{') }
            Some(c) => {
                let span = self.span_from(start, start_line, start_col);
                Err(HoistError::InvalidEscape { ch: c, span })
            }
            None => Err(HoistError::UnterminatedString {
                span: self.span_from(start, start_line, start_col),
            }),
        }
    }

    /// Lex tokens inside a string interpolation `{...}`.
    /// Returns tokens until the matching `}` is found (handling nested braces).
    fn lex_interpolation(&mut self) -> Result<Vec<Token>, HoistError> {
        let mut tokens = Vec::new();
        let mut depth = 1;

        loop {
            self.skip_whitespace_and_comments();
            if self.at_end() {
                return Err(HoistError::SyntaxError {
                    message: "unterminated string interpolation".into(),
                    span: self.current_span(),
                });
            }

            if self.peek() == Some('}') {
                depth -= 1;
                if depth == 0 {
                    self.advance(); // consume closing }
                    break;
                }
            }

            if self.peek() == Some('{') {
                depth += 1;
            }

            let tok = self.next_token()?;
            if tok.kind == TokenKind::LBrace {
                depth += 1; // already advanced
            }
            if tok.kind == TokenKind::RBrace {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            tokens.push(tok);
        }

        Ok(tokens)
    }

    // --- Helper methods ---

    fn peek(&self) -> Option<char> {
        self.source.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<char> {
        self.source.get(self.pos + offset).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.source.get(self.pos).copied()?;
        self.pos += 1;
        if ch == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(ch)
    }

    fn at_end(&self) -> bool {
        self.pos >= self.source.len()
    }

    fn skip_whitespace_and_comments(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.advance();
            } else if c == '-' && self.peek_at(1) == Some('-') {
                self.skip_line_comment();
            } else if c == '{' && self.peek_at(1) == Some('-') {
                self.advance(); // {
                self.advance(); // -
                let _ = self.skip_block_comment();
            } else {
                break;
            }
        }
    }

    fn skip_line_comment(&mut self) {
        while let Some(c) = self.peek() {
            if c == '\n' {
                break;
            }
            self.advance();
        }
    }

    fn skip_block_comment(&mut self) -> Result<(), HoistError> {
        let mut depth = 1;
        while depth > 0 {
            match self.peek() {
                None => {
                    return Err(HoistError::SyntaxError {
                        message: "unterminated block comment".into(),
                        span: self.current_span(),
                    });
                }
                Some('{') if self.peek_at(1) == Some('-') => {
                    self.advance();
                    self.advance();
                    depth += 1;
                }
                Some('-') if self.peek_at(1) == Some('}') => {
                    self.advance();
                    self.advance();
                    depth -= 1;
                }
                _ => {
                    self.advance();
                }
            }
        }
        Ok(())
    }

    fn current_span(&self) -> Span {
        Span::new(self.pos, self.pos, self.line, self.column)
    }

    fn span_from(&self, start: usize, start_line: usize, start_col: usize) -> Span {
        Span::new(start, self.pos, start_line, start_col)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(src: &str) -> Vec<TokenKind> {
        let mut lexer = Lexer::new(src);
        lexer.tokenize()
            .unwrap()
            .into_iter()
            .map(|t| t.kind)
            .filter(|k| *k != TokenKind::Eof)
            .collect()
    }

    #[test]
    fn test_keywords() {
        assert_eq!(lex("let return if then else"), vec![
            TokenKind::Let, TokenKind::Return, TokenKind::If,
            TokenKind::Then, TokenKind::Else,
        ]);
    }

    #[test]
    fn test_identifiers() {
        assert_eq!(lex("foo bar_baz x1"), vec![
            TokenKind::Ident("foo".into()),
            TokenKind::Ident("bar_baz".into()),
            TokenKind::Ident("x1".into()),
        ]);
    }

    #[test]
    fn test_numbers() {
        assert_eq!(lex("42 0 100"), vec![
            TokenKind::Int(42), TokenKind::Int(0), TokenKind::Int(100),
        ]);
    }

    #[test]
    fn test_operators() {
        assert_eq!(lex("|> ++ == != <= >= -> + - * / %"), vec![
            TokenKind::Pipe, TokenKind::PlusPlus, TokenKind::EqEq,
            TokenKind::BangEq, TokenKind::Le, TokenKind::Ge,
            TokenKind::Arrow, TokenKind::Plus, TokenKind::Minus,
            TokenKind::Star, TokenKind::Slash, TokenKind::Percent,
        ]);
    }

    #[test]
    fn test_simple_string() {
        let tokens = lex(r#""hello world""#);
        assert_eq!(tokens.len(), 1);
        match &tokens[0] {
            TokenKind::StringLit(parts) => {
                assert_eq!(parts.len(), 1);
                assert_eq!(parts[0], StringFragment::Literal("hello world".into()));
            }
            _ => panic!("expected string literal"),
        }
    }

    #[test]
    fn test_string_escapes() {
        let tokens = lex(r#""line1\nline2\ttab""#);
        assert_eq!(tokens.len(), 1);
        match &tokens[0] {
            TokenKind::StringLit(parts) => {
                assert_eq!(parts[0], StringFragment::Literal("line1\nline2\ttab".into()));
            }
            _ => panic!("expected string literal"),
        }
    }

    #[test]
    fn test_string_interpolation() {
        let tokens = lex(r#""hello {name}!""#);
        assert_eq!(tokens.len(), 1);
        match &tokens[0] {
            TokenKind::StringLit(parts) => {
                assert_eq!(parts.len(), 3);
                assert_eq!(parts[0], StringFragment::Literal("hello ".into()));
                assert!(matches!(&parts[1], StringFragment::Interpolation(toks) if toks.len() == 1));
                assert_eq!(parts[2], StringFragment::Literal("!".into()));
            }
            _ => panic!("expected string literal"),
        }
    }

    #[test]
    fn test_line_comment() {
        assert_eq!(lex("42 -- this is a comment\n43"), vec![
            TokenKind::Int(42), TokenKind::Int(43),
        ]);
    }

    #[test]
    fn test_block_comment() {
        assert_eq!(lex("42 {- block -} 43"), vec![
            TokenKind::Int(42), TokenKind::Int(43),
        ]);
    }

    #[test]
    fn test_nested_block_comment() {
        assert_eq!(lex("42 {- {- nested -} -} 43"), vec![
            TokenKind::Int(42), TokenKind::Int(43),
        ]);
    }

    #[test]
    fn test_collection_keywords() {
        assert_eq!(lex("map filter fold take drop split join window slice"), vec![
            TokenKind::Map, TokenKind::Filter, TokenKind::Fold,
            TokenKind::Take, TokenKind::Drop, TokenKind::Split,
            TokenKind::Join, TokenKind::Window, TokenKind::Slice,
        ]);
    }

    #[test]
    fn test_ask_keywords() {
        assert_eq!(lex("ask as via fallback"), vec![
            TokenKind::Ask, TokenKind::As, TokenKind::Via, TokenKind::Fallback,
        ]);
    }

    #[test]
    fn test_delimiters() {
        assert_eq!(lex("( ) [ ] { }"), vec![
            TokenKind::LParen, TokenKind::RParen,
            TokenKind::LBracket, TokenKind::RBracket,
            TokenKind::LBrace, TokenKind::RBrace,
        ]);
    }

    #[test]
    fn test_booleans() {
        assert_eq!(lex("true false"), vec![
            TokenKind::True, TokenKind::False,
        ]);
    }

    #[test]
    fn test_unicode_arrow() {
        assert_eq!(lex("\u{2192}"), vec![TokenKind::Arrow]);
    }

    #[test]
    fn test_raw_string() {
        let src = r#""""
raw \ string " here
""""#;
        let tokens = lex(src);
        assert_eq!(tokens.len(), 1);
        match &tokens[0] {
            TokenKind::StringLit(parts) => {
                assert_eq!(parts.len(), 1);
                assert_eq!(parts[0], StringFragment::Literal("raw \\ string \" here".into()));
            }
            _ => panic!("expected string literal"),
        }
    }
}
