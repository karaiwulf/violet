use chrono::Utc;
use reqwest::ClientBuilder;
use serde::Deserialize;
use std::fs::read_to_string;
use std::time::Duration;
use std::io::Write;

use rig::{
    agent::stream_to_stdout,
    prelude::*,
    providers::openai,
    streaming::StreamingChat,
    message::Message,
    client::audio_generation::AudioGenerationClient,
    audio_generation::AudioGenerationModel,
};

#[derive(Deserialize, Clone, Debug)]
struct Config {
    base_url: String,
    key: String,
    model: String,
    audio_model: Option<String>,
    audio_voice: Option<String>,
    system_prompt: String,
    timeout: u64,
    max_tokens: u64,
    temp: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Starting setup");
    eprintln!("Loading Config");
    let config = read_to_string("config.json")?;
    let config: Config = serde_json::from_str(&config)?;
    eprintln!("Config Loaded");
    let conn_timeout = if config.timeout < 30 {
        config.timeout
    } else if config.timeout < 300 {
        config.timeout / 2
    } else {
        config.timeout / 4
    };
    let http_client = ClientBuilder::new()
        .user_agent("violet-rs/0.1")
        .read_timeout(Duration::from_secs(config.timeout))
        .connect_timeout(Duration::from_secs(conn_timeout))
        .build()?;
    let date: String = Utc::now().date_naive().to_string();
    let system_prompt: String = format!("The current date is {date}.  {}", &config.system_prompt);
    eprintln!("System Prompt is: {system_prompt}");
    let api = openai::ClientBuilder::new_with_client(&config.key, http_client)
        .base_url(&config.base_url)
        .build();
    let violet = api.completion_model(&config.model)
        .completions_api()
        .into_agent_builder()
        .preamble(&system_prompt)
        .max_tokens(config.max_tokens)
        .temperature(config.temp)
        .build();
    let audio_model = if let Some(model) = &config.audio_model {
        model
    } else {
        "cosyvoice"
    };
    let _audio_voice = if let Some(voice) = &config.audio_voice {
        voice
    } else {
        "english_female"
    };
    let _violet_voice = api.audio_generation_model(audio_model);
    eprintln!("Base Request Setup");
    eprintln!("Setup Finished");
    let mut s = String::new();
    print!("> ");
    let _ = std::io::stdout().flush();
    if let Err(e) = std::io::stdin().read_line(&mut s) {
        eprintln!("Error reading stdin: {e}");
    }
    let mut history: Vec<Message> = Vec::new();
    let mut uwu = true;
    if "stop" == s.as_str().to_lowercase().trim() {
        uwu = false;
    }
    while uwu {
        let mut stream = violet
            .stream_chat(&s, history.clone())
            .await;
        let res = stream_to_stdout(&mut stream).await?;
        print!("\n");
        //let vres = violet_voice
        //    .audio_generation_request()
        //    .text(res.response())
        //    .voice(audio_voice)
        //    .send()
        //    .await?;
        history.push(Message::user(s.clone()));
        history.push(Message::assistant(res.response()));
        print!("> ");
        s = String::new();
        let _ = std::io::stdout().flush();
        if let Err(e) = std::io::stdin().read_line(&mut s) {
            eprintln!("Error reading stdin: {e}");
        }
        if s.as_str().to_lowercase().trim() == "stop" {
            break;
        }
    }
    Ok(())
}
