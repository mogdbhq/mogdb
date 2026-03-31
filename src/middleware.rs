/// API key authentication middleware.
///
/// Reads `X-Api-Key` header and compares to the configured key.
/// If `API_KEY` env var is not set, all requests are allowed (dev mode).
use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Json, Response},
};

use crate::{models::ErrorBody, AppState};

pub async fn require_api_key(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let Some(ref required) = state.api_key else {
        // No key configured — open access (dev / self-hosted)
        return next.run(request).await;
    };

    let provided = request
        .headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok());

    match provided {
        Some(key) if key == required => next.run(request).await,
        _ => (
            StatusCode::UNAUTHORIZED,
            Json(ErrorBody {
                error: "missing or invalid X-Api-Key".to_string(),
            }),
        )
            .into_response(),
    }
}
