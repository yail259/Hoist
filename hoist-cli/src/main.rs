use std::io::Read;
use hoist_core::{Interpreter, ResourceLimits, Value};
use hoist_core::parser::parse_source;
use hoist_core::interpreter::AskHandler;

const VERSION: &str = env!("CARGO_PKG_VERSION");

struct StdioAskHandler;

impl AskHandler for StdioAskHandler {
    fn ask(&self, prompt: &str, channel: Option<&str>) -> Result<String, String> {
        if let Some(ch) = channel {
            eprintln!("[ask via {}] {}", ch, prompt);
        } else {
            eprintln!("[ask] {}", prompt);
        }
        eprintln!("[ask] Enter response (end with empty line):");
        let mut response = String::new();
        loop {
            let mut line = String::new();
            std::io::stdin().read_line(&mut line).map_err(|e| e.to_string())?;
            if line.trim().is_empty() {
                break;
            }
            response.push_str(&line);
        }
        Ok(response.trim().to_string())
    }
}

fn print_help() {
    println!("hoist {} — A safe language for LLM orchestration", VERSION);
    println!();
    println!("USAGE:");
    println!("    hoist <file.hoist>              Run a Hoist program");
    println!("    hoist --eval '<code>'           Evaluate inline code");
    println!("    cat file.hoist | hoist          Read from stdin");
    println!();
    println!("OPTIONS:");
    println!("    --context <text>                Set the `context` variable");
    println!("    --context-file <path>           Read context from a file");
    println!("    --max-ask-calls <n>             Max LLM ask calls (default: 100)");
    println!("    --max-collection-size <n>       Max collection size (default: 10000)");
    println!("    --max-string-size <n>           Max string size in bytes (default: 10485760)");
    println!("    --max-steps <n>                 Max execution steps (default: 1000000)");
    println!("    --check                         Parse and check only, don't execute");
    println!("    --ast                           Dump the AST (for debugging)");
    println!("    --eval '<code>'                 Evaluate inline code");
    println!("    --version                       Print version");
    println!("    --help                          Print this help");
}

fn get_flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if has_flag(&args, "--help") || has_flag(&args, "-h") {
        print_help();
        return;
    }

    if has_flag(&args, "--version") || has_flag(&args, "-V") {
        println!("hoist {}", VERSION);
        return;
    }

    // Determine source
    let source = if let Some(code) = get_flag_value(&args, "--eval") {
        code
    } else if args.len() > 1 && !args[1].starts_with('-') {
        std::fs::read_to_string(&args[1]).unwrap_or_else(|e| {
            eprintln!("Error reading file '{}': {}", args[1], e);
            std::process::exit(1);
        })
    } else {
        // Read from stdin
        let mut source = String::new();
        std::io::stdin().read_to_string(&mut source).unwrap_or_else(|e| {
            eprintln!("Error reading stdin: {}", e);
            std::process::exit(1);
        });
        source
    };

    // Parse
    let program = match parse_source(&source) {
        Ok(prog) => prog,
        Err(e) => {
            eprintln!("Compile error: {}", e);
            std::process::exit(1);
        }
    };

    // --check: parse only
    if has_flag(&args, "--check") {
        println!("OK");
        return;
    }

    // --ast: dump AST
    if has_flag(&args, "--ast") {
        println!("{:#?}", program);
        return;
    }

    // Configure resource limits
    let mut limits = ResourceLimits::default();
    if let Some(v) = get_flag_value(&args, "--max-ask-calls") {
        limits.max_ask_calls = v.parse().unwrap_or_else(|_| {
            eprintln!("Invalid value for --max-ask-calls: {}", v);
            std::process::exit(1);
        });
    }
    if let Some(v) = get_flag_value(&args, "--max-collection-size") {
        limits.max_collection_size = v.parse().unwrap_or_else(|_| {
            eprintln!("Invalid value for --max-collection-size: {}", v);
            std::process::exit(1);
        });
    }
    if let Some(v) = get_flag_value(&args, "--max-string-size") {
        limits.max_string_size = v.parse().unwrap_or_else(|_| {
            eprintln!("Invalid value for --max-string-size: {}", v);
            std::process::exit(1);
        });
    }
    if let Some(v) = get_flag_value(&args, "--max-steps") {
        limits.max_steps = v.parse().unwrap_or_else(|_| {
            eprintln!("Invalid value for --max-steps: {}", v);
            std::process::exit(1);
        });
    }

    // Set up interpreter
    let mut interp = Interpreter::new()
        .with_ask_handler(Box::new(StdioAskHandler))
        .with_limits(limits);

    // Context from --context or --context-file
    if let Some(ctx) = get_flag_value(&args, "--context") {
        interp.set_var("context", Value::String(ctx));
    } else if let Some(path) = get_flag_value(&args, "--context-file") {
        let ctx = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!("Error reading context file '{}': {}", path, e);
            std::process::exit(1);
        });
        interp.set_var("context", Value::String(ctx));
    }

    // Execute
    match interp.eval_program(&program) {
        Ok(result) => println!("{}", result.display()),
        Err(e) => {
            eprintln!("Runtime error: {}", e);
            std::process::exit(1);
        }
    }
}
