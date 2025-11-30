use base64::{Engine, engine::general_purpose::STANDARD};
use chrono::Utc;
use reqwest::{Client, ClientBuilder};
use rig::{
  agent::{Agent, stream_to_stdout},
  audio_generation::AudioGenerationModel,
  client::audio_generation::AudioGenerationClient,
  completion::Chat,
  message::{
    AssistantContent, DocumentSourceKind, Image, ImageDetail, ImageMediaType,
    Message, UserContent,
  },
  prelude::*,
  providers::openai,
  providers::openai::CompletionModel,
  streaming::StreamingChat,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io::{Cursor, Write};
use std::time::Duration;

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Config {
  base_url: String,
  key: String,
  timeout: u64,
  #[serde(skip_serializing_if = "Option::is_none")]
  vision_model: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  vision_prompt: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  summary_model: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  summary_prompt: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  audio_model: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  audio_voice: Option<String>,
  vision_tokens: u64,
  vision_temp: f64,
  summary_tokens: u64,
  summary_temp: f64,
}
impl std::default::Default for Config {
  fn default() -> Self {
    Self {
      base_url: String::from("https://api.openai.com/v1"),
      key: String::from("sk-..."),
      vision_model: None,
      vision_prompt: None,
      summary_prompt: None,
      summary_model: None,
      audio_model: None,
      audio_voice: None,
      timeout: 30,
      vision_tokens: 4096,
      vision_temp: 0.4,
      summary_tokens: 8192,
      summary_temp: 0.9,
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
    confy::get_configuration_file_path("violet", Some("violet"))?
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
  let vision_prompt: String = if let Some(prompt) = config.vision_prompt {
    prompt
  } else {
    "You will describe the images attached".into()
  };
  let summary_prompt: String = format!(
    "The current date is {date}.\n\n{}",
    if let Some(prompt) = config.summary_prompt {
      prompt
    } else {
      String::from("You will create a narrative for the image ")
        + "descriptions given as if you were telling a story."
    }
  );
  eprintln!("Vision System Prompt is: {vision_prompt}");
  eprintln!("Summary System Prompt is: {summary_prompt}");
  let api = openai::ClientBuilder::new_with_client(&config.key, http_client)
    .base_url(&config.base_url)
    .build();
  let vision_model: String = if let Some(vmodel) = config.vision_model {
    vmodel
  } else {
    "gpt-image-1".into()
  };
  let vision = api
    .completion_model(&vision_model)
    .completions_api()
    .into_agent_builder()
    .preamble(&vision_prompt)
    .max_tokens(config.vision_tokens)
    .temperature(config.vision_temp)
    .build();
  let summary_model: String = if let Some(smodel) = config.summary_model {
    smodel
  } else {
    "gpt-4o".into()
  };
  let summary = api
    .completion_model(&summary_model)
    .completions_api()
    .into_agent_builder()
    .preamble(&summary_prompt)
    .max_tokens(config.summary_tokens)
    .temperature(config.summary_temp)
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
  let audio = api.audio_generation_model(audio_model);
  eprintln!("Setup Finished");
  routing(vision, summary, audio, audio_voice).await?;
  Ok(())
}

async fn chat(
  agent: Agent<CompletionModel<Client>>,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
  let mut history: Vec<Message> = Vec::new();
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
    let mut stream = agent.stream_chat(&s, history.clone()).await;
    let res = stream_to_stdout(&mut stream).await?;
    print!("\n");
    history.push(Message::user(s.clone()));
    history.push(Message::assistant(res.response()));
    print!("> ");
    s = String::new();
    let _ = std::io::stdout().flush();
    if let Err(e) = std::io::stdin().read_line(&mut s) {
      eprintln!("Error reading stdin: {e}");
    }
    if s.as_str().to_lowercase().trim() == "stop" {
      uwu = false;
    }
  }
  Ok(history)
}

async fn prompt_model(
  agent: Agent<CompletionModel<Client>>,
  prompt: Message,
  history: Vec<Message>,
) -> Result<Message, Box<dyn std::error::Error>> {
  let res = agent.chat(prompt, history).await?;
  Ok(rig::message::AssistantContent::text(&res).into())
}

async fn get_audio(
  audio: openai::audio_generation::AudioGenerationModel,
  voice: &str,
  text: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
  let vres = audio
    .audio_generation_request()
    .text(text)
    .voice(voice)
    .additional_params(json!(
      {
        "response_format": "mp3",
      }
    ))
    .send()
    .await?;
  Ok(vres.audio.clone())
}

async fn routing(
  vision: Agent<CompletionModel<Client>>,
  summary: Agent<CompletionModel<Client>>,
  audio: openai::audio_generation::AudioGenerationModel,
  audio_voice: &str,
) -> Result<(), Box<dyn std::error::Error>> {
  let _vision = vision;
  let mut s: String = String::new();
  for m in chat(summary).await? {
    let text: String = match m {
      Message::User { content } => {
        let mut e: String = "User: ".into();
        for c in content {
          if let UserContent::Text(content) = c {
            e = e + content.text().into();
            e = e + "\n".into();
          }
        }
        e
      },
      Message::Assistant { id, content } => {
        let _id = id;
        let mut e: String = "Assistant: ".into();
        for c in content {
          if let AssistantContent::Text(content) = c {
            e = e + content.text().into();
            e = e + "\n".into();
          }
        }
        e
      },
    };
    s = s + &text;
  }
  let e = get_audio(audio, audio_voice, &s).await?;
  let mut fiel = std::fs::OpenOptions::new()
    .create(true)
    .write(true)
    .truncate(true)
    .open("chat.mp3")?;
  fiel.write_all(&e.as_slice())?;
  Ok(())
}
