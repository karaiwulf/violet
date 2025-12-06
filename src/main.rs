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
use surrealdb::{
  Surreal,
  engine::local::RocksDb,
};

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ModelOptions {
  #[serde(skip_serializing_if = "Option::is_none")]
  model: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  prompt: Option<String>,
  tokens: u64,
  temp: f64,
}
impl std::default::Default for ModelOptions {
  fn default() -> Self {
    Self {
      model: None,
      prompt: None,
      tokens: 4096,
      temp: 1.0,
    }
  }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct TTSOptions {
  model: String,
  voice: String,
}
impl std::default::Default for TTSOptions {
  fn default() -> Self {
    Self {
      model: String::from("tts-1"),
      voice: String::from("Alloy"),
    }
  }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ApiOptions {
  base_url: String,
  key: String,
  timeout: u64,
}
impl std::default::Default for ApiOptions {
  fn default() -> Self {
  Self {
    base_url: String::from("https://api.openai.com/v1"),
    key: String::from("sk-..."),
    timeout: 30,
    }
  }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Config {
  api: ApiOptions,
  vision: ModelOptions,
  summary: ModelOptions,
  #[serde(skip_serializing_if = "Option::is_none")]
  tts: Option<TTSOptions>,
}
impl std::default::Default for Config {
  fn default() -> Self {
    Self {
      api: ApiOptions::default(),
      vision: ModelOptions::default(),
      summary: ModelOptions::default(),
      tts: None,
    }
  }
}

#[derive(Debug, Serialize, Deserialize)]
struct Record {
  id: surrealdb::RecordId,
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
  let conn_timeout = if config.api.timeout < 30 {
    config.api.timeout
  } else if config.api.timeout < 300 {
    config.api.timeout / 2
  } else {
    config.api.timeout / 4
  };
  let http_client = ClientBuilder::new()
    .user_agent("violet-rs/0.1")
    .read_timeout(Duration::from_secs(config.api.timeout))
    .connect_timeout(Duration::from_secs(conn_timeout))
    .build()?;
  let date: String = Utc::now().date_naive().to_string();
  let vision_prompt: String = if let Some(prompt) = config.vision.prompt {
    prompt
  } else {
    "You will describe the images attached".into()
  };
  let summary_prompt: String = format!(
    "The current date is {date}.\n\n{}",
    if let Some(prompt) = config.summary.prompt {
      prompt
    } else {
      String::from("You will create a narrative for the image ")
        + "descriptions given as if you were telling a story."
    }
  );
  eprintln!("Vision System Prompt is: {vision_prompt}");
  eprintln!("Summary System Prompt is: {summary_prompt}");
  let api = openai::ClientBuilder::new_with_client(&config.api.key, http_client)
    .base_url(&config.api.base_url)
    .build();
  let vision_model: String = if let Some(vmodel) = config.vision.model {
    vmodel
  } else {
    "gpt-image-1".into()
  };
  let vision = api
    .completion_model(&vision_model)
    .completions_api()
    .into_agent_builder()
    .preamble(&vision_prompt)
    .max_tokens(config.vision.tokens)
    .temperature(config.vision.temp)
    .build();
  let summary_model: String = if let Some(smodel) = config.summary.model {
    smodel
  } else {
    "gpt-4o".into()
  };
  let summary = api
    .completion_model(&summary_model)
    .completions_api()
    .into_agent_builder()
    .preamble(&summary_prompt)
    .max_tokens(config.summary.tokens)
    .temperature(config.summary.temp)
    .build();
  let (audio_model, audio_voice) = if let Some(tts) = &config.tts {
      (tts.model.as_str(), tts.voice.as_str())
  } else {
      ("tts-1", "Alloy")
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
  Ok(AssistantContent::text(&res).into())
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
  let db = Surreal::new::<RocksDb>("db").await?;
  db.use_ns("violet").use_db("manga").await?;
  /*TODO:
   * 1. pick manga based on tags via mangadex api.
   * 2. batch and send pages to gemma-3 VL model
   * 3. send descriptions to gpt-oss for verification
   * 4. send approved descriptions back to gpt-oss for narrative generation
   * 5. send gpt-oss output to kokoro
   * 6. potentially get gemma-3 to mark important frames for video purposes?
   * 7. compile video from template, images, and kokoro output
   * 8. upload directly to youtube via some youtube api crate
   */
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
