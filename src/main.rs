use chrono::Utc;
use reqwest::ClientBuilder;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use std::io::{Write, Cursor};
//use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::json;
use rodio::Decoder;

use rig::{
    agent::stream_to_stdout,
    prelude::*,
    providers::openai,
    streaming::StreamingChat,
    message::{Message, Image, ImageMediaType, DocumentSourceKind, ImageDetail},
    client::audio_generation::AudioGenerationClient,
    audio_generation::AudioGenerationModel,
};

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Config {
    base_url: String,
    key: String,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_voice: Option<String>,
    system_prompt: String,
    timeout: u64,
    max_tokens: u64,
    temp: f64,
}
impl std::default::Default for Config {
    fn default() -> Self {
        Self {
            base_url: String::from("https://api.openai.com/v1"),
            key: String::from("sk-..."),
            model: String::from("gpt-4o"),
            audio_model: None,
            audio_voice: None,
            system_prompt: String::from("You are a helpful assistant!"),
            timeout: 30,
            max_tokens: 4096,
            temp: 0.4,
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Starting setup");
    eprintln!("Loading Config");
    let config: Config = confy::load("violet", Some("violet"))?;
    println!(
        "Config file location: {}",
        confy::get_configuration_file_path("violet", None)?
            .as_path()
            .to_str()
            .unwrap_or("path does not exist")
    );
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
    let system_prompt: String = format!("The current date is {date}.\n\n{}", &config.system_prompt);
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
        "tts-1"
    };
    let audio_voice = if let Some(voice) = &config.audio_voice {
        voice
    } else {
        "alloy"
    };
    let violet_voice = api.audio_generation_model(audio_model);
    eprintln!("Base Request Setup");
    let mut history: Vec<Message> = Vec::new();
    eprintln!("Getting Audio Device");
    let stream_handle = rodio::OutputStreamBuilder::open_default_stream()?;
    let _sink = rodio::Sink::connect_new(&stream_handle.mixer());
    eprintln!("Setup Finished");
    let mut s = String::new();
    print!("> ");
    let _ = std::io::stdout().flush();
    if let Err(e) = std::io::stdin().read_line(&mut s) {
        eprintln!("Error reading stdin: {e}");
    }
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
        let vres = violet_voice
            .audio_generation_request()
            .text(res.response())
            .voice(audio_voice)
            .additional_params(json!({"response_format": "mp3"}))
            .send()
            .await?;
        let vdata = Decoder::new(Cursor::new(vres.audio.clone()))?;
        stream_handle.mixer().add(vdata);
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
