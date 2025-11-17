use llm_connector::{LlmClient, types::{ChatRequest, Message, Role, Tool, Function}};
use serde_json::{json, Value};
use serde::Deserialize;
use std::fs::read_to_string;
use std::io::Write;
use chrono::Utc;

#[derive(Deserialize, Clone, Debug)]
struct Config {
    base_url: String,
    key: String,
    model: String,
    timeout: u64,
}

async fn get_horoscope(sign: &str) -> String {
    format!("{sign}: Next Tuesday you will befriend a baby otter.")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Starting setup");
    eprintln!("Loading Config");
    let config = read_to_string("config.json")?;
    let config: Config = serde_json::from_str(&config)?;
    eprintln!("Config Loaded");
    let client = LlmClient::openai_with_config(
        &config.key,
        Some(&config.base_url),
        Some(config.timeout),
        None,
    )?;
    eprintln!("Config Setup");
    let mut tools: Vec<Tool> = Vec::new();
    tools.push(Tool {
        tool_type: "function".into(),
        function: Function {
            name: "get_horoscope".into(),
            description: Some("Get today's horoscope for an astrological sign.".into()),
            parameters: json!({
                "sign": {
                    "type": "string",
                    "description": "An astrological sign like Taurus or Aquarius."
                }
            }),
        },
    });
    tools.push(Tool {
        tool_type: "function".into(),
        function: Function {
            name: "stop".into(),
            description: Some("Emergency Stop the Conversation".into()),
            parameters: json!({}),
        },
    });
    eprintln!("Tools Loaded");
    let date: String = Utc::now().date_naive().to_string();
    let system_prompt: String = format!("You are a helpful agent!  You will comply with all user requests.  The current date is {date}.");
    let mut req = ChatRequest {
        model: config.model,
        messages: vec![
            Message::text(Role::System, &system_prompt),
        ],
        tools: Some(tools),
        //temperature: Some(0.9),
        ..Default::default()
    };
    eprintln!("System Prompt is: {system_prompt}");
    eprintln!("Base Request Setup");
    eprintln!("Setup Finished");
    loop {
        let mut s = String::new();
        print!("user> ");
        std::io::stdout().flush()?;
        if let Err(e) = std::io::stdin().read_line(&mut s) {
            eprintln!("could not read stdin: {e}");
            break;
        }
        if s == "stop" || s == "quit" {
            break;
        }
        req.messages.push(Message::text(Role::User, &s));
        let mut response = client.chat(&req).await?;
        for choice in response.choices.clone() {
            if let Some(reason) = choice.finish_reason {
                match reason.as_str() {
                    "tool_calls" => {
                        if let Some(calls) = choice.message.clone().tool_calls {
                            for call in calls {
                                match call.function.name.as_str() {
                                    "get_horoscope" => {
                                        let v: Value = serde_json::from_str(call.function.arguments.as_str())?;
                                        req.messages.push(choice.message.clone());
                                        let v = v["sign"].as_str().unwrap_or_default();
                                        let val: String = get_horoscope(v).await;
                                        req.messages.push(Message::tool(val, call.id));
                                        response = client.chat(&req).await?;
                                    },
                                    "stop" => {
                                        println!("Agent has stopped the conversation.");
                                        if let Some(reasoning) = response.reasoning_content {
                                            eprintln!("Agent Stopped with Reasoning: {}", reasoning.as_str().trim_start().trim_end());
                                        } else {
                                            eprintln!("Agent has not given reasoning for emergency stop.");
                                        }
                                        return Ok(());
                                    }
                                    _ => (),
                                }
                            }
                        }
                    },
                    _ => (),
                }
            }
        }
        println!("agent> {}", &response.content.trim_start().trim_end());
        req.messages.push(Message::text(Role::Assistant, &response.content.clone()));
    }
    Ok(())
}
