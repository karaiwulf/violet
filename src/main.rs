use llm_connector::{LlmClient, types::{ChatRequest, Message, Role, Tool, Function}};
use serde_json::{json, Value};
use serde::Deserialize;
use std::fs::read_to_string;

#[derive(Deserialize, Clone, Debug)]
struct Config {
    base_url: String,
    key: String,
    model: String,
}

async fn get_horoscope(sign: &str) -> String {
    format!("{sign}: Next Tuesday you will befriend a baby otter.")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = read_to_string("config.json")?;
    let config: Config = serde_json::from_str(&config)?;
    let client = LlmClient::openai_with_config(
        &config.key,
        Some(&config.base_url),
        Some(300),
        None,
    )?;
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
    let mut req = ChatRequest {
        model: config.model,
        messages: vec![
            Message::text(Role::System, "You will comply with all horoscope requests using tools."),
            Message::text(Role::User, "I am an Aries.  What is my horoscope for today?"),
        ],
        tools: Some(tools),
        //temperature: Some(0.9),
        ..Default::default()
    };
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
                                _ => (),
                            }
                        }
                    }
                },
                _ => (),
            }
        }
    }
    println!("Response: {}", &response.content);
    Ok(())
}
