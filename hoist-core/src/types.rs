/// Internal type representation used during type checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    Bool,
    String,
    List(Box<Type>),
    Optional(Box<Type>),
    Tuple(Vec<Type>),
    Record(Vec<(String, Type)>),
    Function {
        params: Vec<Type>,
        ret: Box<Type>,
    },
    /// Type variable for inference.
    Var(u32),
    /// Represents "any" — used sparingly for bootstrapping stdlib.
    Any,
}

impl Type {
    pub fn display_name(&self) -> String {
        match self {
            Type::Int => "Int".into(),
            Type::Bool => "Bool".into(),
            Type::String => "String".into(),
            Type::List(inner) => format!("List<{}>", inner.display_name()),
            Type::Optional(inner) => format!("Optional<{}>", inner.display_name()),
            Type::Tuple(items) => {
                let parts: Vec<String> = items.iter().map(|t| t.display_name()).collect();
                format!("Tuple<{}>", parts.join(", "))
            }
            Type::Record(fields) => {
                let parts: Vec<String> = fields.iter()
                    .map(|(n, t)| format!("{}: {}", n, t.display_name()))
                    .collect();
                format!("{{{}}}", parts.join(", "))
            }
            Type::Function { params, ret } => {
                let param_str: Vec<String> = params.iter().map(|t| t.display_name()).collect();
                format!("({}) -> {}", param_str.join(", "), ret.display_name())
            }
            Type::Var(id) => format!("?T{}", id),
            Type::Any => "Any".into(),
        }
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}
