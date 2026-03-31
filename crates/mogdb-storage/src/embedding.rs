/// Embedding providers — generate dense vector representations of text.
///
/// The trait uses RPITIT (return-position impl Trait in trait, stable since Rust 1.75)
/// so no async-trait dependency is needed. Use with generics, not dyn.
use mogdb_core::MogError;
use std::future::Future;

/// A source that can produce a dense embedding vector for a piece of text.
pub trait EmbeddingProvider: Send + Sync {
    fn embed(&self, text: &str) -> impl Future<Output = Result<Vec<f32>, MogError>> + Send + '_;
}

/// Ollama embeddings — runs local models, zero API key required.
///
/// Default model: `mxbai-embed-large` (1024 dims, MTEB 64.68, 1.3 GB).
/// Other good options:
///   - `nomic-embed-text`   — 768 dims, 274 MB, MTEB 62.39 (lightest)
///   - `bge-m3`             — 1024 dims, 1.2 GB, excellent multilingual
///
/// Pull a model first: `ollama pull mxbai-embed-large`
///
/// The default base URL is `http://localhost:11434`. Override with `OLLAMA_HOST`.
#[derive(Clone)]
pub struct OllamaEmbeddings {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl OllamaEmbeddings {
    /// Connect to Ollama at `http://localhost:11434` using `mxbai-embed-large`.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "http://localhost:11434".to_string(),
            model: "mxbai-embed-large".to_string(),
        }
    }

    /// Reads `OLLAMA_HOST` env var, falls back to `http://localhost:11434`.
    pub fn from_env() -> Self {
        let base_url =
            std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
        Self {
            client: reqwest::Client::new(),
            base_url,
            model: "mxbai-embed-large".to_string(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

impl Default for OllamaEmbeddings {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingProvider for OllamaEmbeddings {
    fn embed(&self, text: &str) -> impl Future<Output = Result<Vec<f32>, MogError>> + Send + '_ {
        let text = text.to_owned();
        async move {
            // Ollama /api/embed endpoint (v0.1.31+)
            // Docs: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
            #[derive(serde::Serialize)]
            struct EmbedRequest {
                model: String,
                input: String,
            }

            #[derive(serde::Deserialize)]
            struct EmbedResponse {
                embeddings: Vec<Vec<f32>>,
            }

            let url = format!("{}/api/embed", self.base_url);

            let resp = self
                .client
                .post(&url)
                .json(&EmbedRequest {
                    model: self.model.clone(),
                    input: text,
                })
                .send()
                .await
                .map_err(|e| {
                    MogError::External(format!(
                        "Ollama request failed (is Ollama running at {}?): {e}",
                        self.base_url
                    ))
                })?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(MogError::External(format!(
                    "Ollama embed API {status}: {body}"
                )));
            }

            let body: EmbedResponse = resp.json().await.map_err(|e| {
                MogError::External(format!("failed to parse Ollama embed response: {e}"))
            })?;

            body.embeddings
                .into_iter()
                .next()
                .ok_or_else(|| MogError::External("empty embeddings array from Ollama".to_string()))
        }
    }
}
