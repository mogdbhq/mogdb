/// LLM provider trait and Ollama implementation.
///
/// Used by the ConsolidationWorker to summarise clusters of episodic memories
/// into a single semantic memory.
///
/// Default model: `llama3.2` (4-bit quantised, ~2 GB, good instruction following).
/// Override with `OLLAMA_LLM_MODEL` and `OLLAMA_HOST` env vars.
use crate::error::WorkerError;
use std::future::Future;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A language model that can complete a prompt.
pub trait LlmProvider: Send + Sync {
    fn complete(
        &self,
        prompt: &str,
    ) -> impl Future<Output = Result<String, WorkerError>> + Send + '_;
}

// ---------------------------------------------------------------------------
// OllamaLlm
// ---------------------------------------------------------------------------

pub struct OllamaLlm {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl OllamaLlm {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "http://localhost:11434".to_string(),
            model: "llama3.2".to_string(),
        }
    }

    /// Reads `OLLAMA_HOST` and `OLLAMA_LLM_MODEL` env vars; falls back to defaults.
    pub fn from_env() -> Self {
        let base_url =
            std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
        let model = std::env::var("OLLAMA_LLM_MODEL").unwrap_or_else(|_| "llama3.2".to_string());
        Self {
            client: reqwest::Client::new(),
            base_url,
            model,
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

impl Default for OllamaLlm {
    fn default() -> Self {
        Self::new()
    }
}

impl LlmProvider for OllamaLlm {
    fn complete(
        &self,
        prompt: &str,
    ) -> impl Future<Output = Result<String, WorkerError>> + Send + '_ {
        let prompt = prompt.to_owned();
        async move {
            #[derive(serde::Serialize)]
            struct GenerateRequest {
                model: String,
                prompt: String,
                stream: bool,
            }

            #[derive(serde::Deserialize)]
            struct GenerateResponse {
                response: String,
            }

            let url = format!("{}/api/generate", self.base_url);

            let resp = self
                .client
                .post(&url)
                .json(&GenerateRequest {
                    model: self.model.clone(),
                    prompt,
                    stream: false,
                })
                .send()
                .await
                .map_err(|e| {
                    WorkerError::Llm(format!(
                        "Ollama request failed (is Ollama running at {}?): {e}",
                        self.base_url
                    ))
                })?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(WorkerError::Llm(format!(
                    "Ollama generate API {status}: {body}"
                )));
            }

            let body: GenerateResponse = resp.json().await.map_err(|e| {
                WorkerError::Llm(format!("failed to parse Ollama generate response: {e}"))
            })?;

            Ok(body.response.trim().to_string())
        }
    }
}
