mod agent;
mod decode;
mod model;
mod sequence;

use anyhow::Result;
use candle_core::Device;
use clap::Parser;
use serde_json::Value;

use agent::Agent;

#[derive(Parser, Debug)]
#[command(name = "laya-candle", about = "English Laya inference with Candle")]
struct Args {
    /// Hub id or local directory containing rl_agent_config.json
    #[arg(long, default_value = "convaiinnovations/laya")]
    model: String,

    /// State text (or JSON via --state-json)
    #[arg(long)]
    state: Option<String>,

    /// Path to a JSON file with the state
    #[arg(long)]
    state_json: Option<String>,

    /// Path to questions JSON object
    #[arg(long)]
    questions: Option<String>,

    /// Run the sample from model.py
    #[arg(long, default_value_t = false)]
    demo: bool,
}

fn default_demo() -> (Value, Value) {
    let state = Value::String(
        "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
            .into(),
    );
    let questions = serde_json::json!({
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs, outages, system errors",
                "other": "everything else"
            }
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["not urgent", "soon", "blocking"]
        },
        "churn_risk": {
            "type": "noul",
            "instructions": "Does the user threaten to cancel or leave?"
        }
    });
    (state, questions)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (state, questions) = if args.demo || (args.state.is_none() && args.state_json.is_none()) {
        default_demo()
    } else {
        let state = if let Some(path) = &args.state_json {
            serde_json::from_str(&std::fs::read_to_string(path)?)?
        } else {
            Value::String(args.state.clone().unwrap_or_default())
        };
        let Some(path) = &args.questions else {
            anyhow::bail!("--questions PATH is required unless --demo");
        };
        let questions = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        (state, questions)
    };

    eprintln!("loading model {} ...", args.model);
    let agent = Agent::load(&args.model, Device::Cpu)?;
    eprintln!("running system_one ...");
    let result = agent.system_one(&state, &questions)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
