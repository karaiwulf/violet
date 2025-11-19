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
    system_prompt: String,
    timeout: u64,
}

async fn get_horoscope(sign: &str) -> String {
    format!("{sign}: Next Tuesday you will befriend a baby otter.")
}

async fn wikipedia_lookup(title: &str) -> String {
    format!("{title}: This article is the article on {title} and is quite interesting.  {title} has several toes.  Many more toes than a {title} should have.")
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
            description: Some("Emergency Stop the Conversation.  Only to be used when the user is requesting something dangerous.".into()),
            parameters: json!({}),
        },
    });
    tools.push(Tool {
        tool_type: "function".into(),
        function: Function {
            name: "wikipedia_lookup".into(),
            description: Some("Look up a wikipedia article and have its summary returned.".into()),
            parameters: json!({
                "title": {
                    "type": "string",
                    "description": "The title of the article to look up."
                }
            }),
        },
    });
    eprintln!("Tools Loaded");
    let date: String = Utc::now().date_naive().to_string();
    let system_prompt: String = format!("The current date is {date}.  {}", &config.system_prompt);
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
    let mut s = String::new();
    print!("user> ");
    std::io::stdout().flush()?;
    if let Err(e) = std::io::stdin().read_line(&mut s) {
        eprintln!("could not read stdin: {e}");
        return Ok(())
    }
    if s.as_str().trim().to_lowercase() == "stop" || s.as_str().trim().to_lowercase() == "quit" {
        return Ok(());
    }
    req.messages.push(Message::text(Role::User, &s));
    let mut uwu: bool = true;
    while uwu {
        let response = client.chat(&req).await?;
        eprintln!("{:?}", &response);
        for choice in response.choices.clone() {
            if let Some(reason) = choice.finish_reason {
                match reason.as_str() {
                    "tool_calls" => {
                        if let Some(calls) = choice.message.clone().tool_calls {
                            for call in calls {
                                match call.function.name.as_str() {
                                    "get_horoscope" => {
                                        eprintln!("get_horoscope!");
                                        let v: Value = serde_json::from_str(call.function.arguments.as_str())?;
                                        req.messages.push(choice.message.clone());
                                        let v = v["sign"].as_str().unwrap_or_default();
                                        let val: String = get_horoscope(v).await;
                                        req.messages.push(Message::tool(val, call.id));
                                        eprintln!("{:?}", &req);
                                    },
                                    "wikipedia_lookup" => {
                                        eprintln!("wikipedia_lookup!");
                                        let v: Value = serde_json::from_str(call.function.arguments.as_str())?;
                                        req.messages.push(choice.message.clone());
                                        let v = v["title"].as_str().unwrap_or_default();
                                        let val: String = wikipedia_lookup(v).await;
                                        req.messages.push(Message::tool(val, call.id));
                                        eprintln!("{:?}", &req);
                                    }
                                    "stop" => {
                                        println!("Agent has stopped the conversation.");
                                        if let Some(reasoning) = response.reasoning_content {
                                            eprintln!("Agent Stopped with Reasoning: {}", reasoning.as_str().trim());
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
                    "stop" => {
                        println!("agent> {}", &response.content.trim());
                        req.messages.push(Message::text(Role::Assistant, &response.content.clone()));
                        let mut s = String::new();
                        print!("user> ");
                        std::io::stdout().flush()?;
                        if let Err(e) = std::io::stdin().read_line(&mut s) {
                            eprintln!("could not read stdin: {e}");
                            uwu = false;
                            break;
                        }
                        if s.as_str().trim().to_lowercase() == "stop" || s.as_str().trim().to_lowercase() == "quit" {
                            uwu = false;
                            break;
                        }
                        req.messages.push(Message::text(Role::User, &s));
                    },
                    _ => (),
                }
            }
        }
    }
    Ok(())
}
