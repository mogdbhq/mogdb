/// Axum route handlers for the MogDB REST API.
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    middleware,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use metrics::counter;
use mogdb_core::MemoryKind;
use mogdb_storage::{
    entity, memory, pipeline, reranking, search::SearchQuery, search_hybrid, OllamaEmbeddings,
};
use uuid::Uuid;

use crate::{
    middleware::require_api_key,
    models::{
        parse_kind, parse_source_trust, ErrorBody, GraphData, GraphEdge, GraphNode, IngestBody,
        IngestResponse, SearchParams,
    },
    AppState,
};

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

pub fn build_router(state: AppState) -> Router {
    // Routes that require API key auth
    let protected = Router::new()
        .route("/memories", post(create_memory))
        .route("/memories/search", get(search_memories))
        .route("/memories/{id}", get(get_memory).delete(expire_memory))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            require_api_key,
        ));

    // Public routes — no auth required
    let public = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(prometheus_metrics))
        .route("/explorer", get(explorer))
        .route("/api/graph", get(graph_data));

    Router::new()
        .merge(protected)
        .merge(public)
        .with_state(state)
}

// ---------------------------------------------------------------------------
// POST /memories
// ---------------------------------------------------------------------------

async fn create_memory(
    State(state): State<AppState>,
    Json(body): Json<IngestBody>,
) -> impl IntoResponse {
    let kind = match parse_kind(body.kind.as_deref().unwrap_or("episodic")) {
        Ok(k) => k,
        Err(e) => return bad_request(&e),
    };
    let source_trust = match parse_source_trust(body.source_trust.as_deref().unwrap_or("agent")) {
        Ok(t) => t,
        Err(e) => return bad_request(&e),
    };

    let embedder = OllamaEmbeddings::from_env();

    match pipeline::ingest_with_embedder(
        &state.pool,
        &body.agent_id,
        &body.user_id,
        &body.content,
        kind,
        source_trust,
        body.session_id.as_deref(),
        &embedder,
    )
    .await
    {
        Ok(result) => {
            counter!("mogdb_api_memories_created_total").increment(1);
            (
                StatusCode::CREATED,
                Json(IngestResponse {
                    id: result.memory.id,
                    importance: result.memory.importance,
                    conflicts_invalidated: result.conflicts_invalidated,
                    entities_touched: result.entities_touched,
                    quarantined: result.quarantined,
                }),
            )
                .into_response()
        }
        Err(e) => internal_error(&e.to_string()),
    }
}

// ---------------------------------------------------------------------------
// GET /memories/search
// ---------------------------------------------------------------------------

async fn search_memories(
    State(state): State<AppState>,
    Query(params): Query<SearchParams>,
) -> impl IntoResponse {
    let kind: Option<MemoryKind> = match params.kind.as_deref() {
        Some(k) => match parse_kind(k) {
            Ok(kind) => Some(kind),
            Err(e) => return bad_request(&e),
        },
        None => None,
    };

    let mut query = SearchQuery::new(&params.agent_id, &params.user_id, &params.q)
        .limit(params.limit.unwrap_or(10));

    if let Some(k) = kind {
        query = query.kind(k);
    }
    if let Some(ts) = params.as_of {
        query = query.as_of(ts);
    }
    if let Some(min) = params.min_strength {
        query = query.min_strength(min);
    }
    if params.include_graph.unwrap_or(false) {
        query = query.with_graph();
    }

    let embedder = OllamaEmbeddings::from_env();

    match search_hybrid(&state.pool, query, &embedder).await {
        Ok(results) => {
            let results = reranking::rerank(&params.q, results).await;
            counter!("mogdb_api_searches_total").increment(1);
            Json(results).into_response()
        }
        Err(e) => internal_error(&e.to_string()),
    }
}

// ---------------------------------------------------------------------------
// GET /memories/:id
// ---------------------------------------------------------------------------

async fn get_memory(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Query(params): Query<AgentQuery>,
) -> impl IntoResponse {
    match memory::fetch_by_id(&state.pool, id, &params.agent_id).await {
        Ok(record) => Json(record).into_response(),
        Err(mogdb_core::MogError::NotFound(_)) => not_found(&id.to_string()),
        Err(e) => internal_error(&e.to_string()),
    }
}

// ---------------------------------------------------------------------------
// DELETE /memories/:id
// ---------------------------------------------------------------------------

async fn expire_memory(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Query(params): Query<AgentQuery>,
) -> impl IntoResponse {
    match memory::expire(&state.pool, id, &params.agent_id).await {
        Ok(()) => {
            counter!("mogdb_api_memories_expired_total").increment(1);
            StatusCode::NO_CONTENT.into_response()
        }
        Err(mogdb_core::MogError::NotFound(_)) => not_found(&id.to_string()),
        Err(e) => internal_error(&e.to_string()),
    }
}

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    match sqlx::query("SELECT 1").execute(&state.pool).await {
        Ok(_) => Json(serde_json::json!({"status": "ok", "db": "up"})).into_response(),
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"status": "degraded", "error": e.to_string()})),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// GET /metrics
// ---------------------------------------------------------------------------

async fn prometheus_metrics(State(state): State<AppState>) -> impl IntoResponse {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        state.prom_handle.render(),
    )
}

// ---------------------------------------------------------------------------
// Shared query params
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct AgentQuery {
    agent_id: String,
}

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

fn bad_request(msg: &str) -> axum::response::Response {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorBody {
            error: msg.to_string(),
        }),
    )
        .into_response()
}

fn not_found(id: &str) -> axum::response::Response {
    (
        StatusCode::NOT_FOUND,
        Json(ErrorBody {
            error: format!("memory {id} not found"),
        }),
    )
        .into_response()
}

fn internal_error(msg: &str) -> axum::response::Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorBody {
            error: msg.to_string(),
        }),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// GET /explorer
// ---------------------------------------------------------------------------

async fn explorer() -> impl IntoResponse {
    axum::response::Html(include_str!("explorer.html"))
}

// ---------------------------------------------------------------------------
// GET /api/graph
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct GraphQuery {
    agent_id: String,
    user_id: String,
}

async fn graph_data(
    State(state): State<AppState>,
    Query(params): Query<GraphQuery>,
) -> impl IntoResponse {
    let (memories_res, entities_res, edges_res) = tokio::join!(
        memory::list_active(&state.pool, &params.agent_id, &params.user_id, 300),
        entity::list_all(&state.pool, &params.agent_id, &params.user_id),
        entity::list_all_edges(&state.pool, &params.agent_id, &params.user_id),
    );

    let memories = match memories_res {
        Ok(m) => m,
        Err(e) => return internal_error(&e.to_string()),
    };
    let entities = match entities_res {
        Ok(e) => e,
        Err(e) => return internal_error(&e.to_string()),
    };
    let edges = match edges_res {
        Ok(e) => e,
        Err(e) => return internal_error(&e.to_string()),
    };

    // Build a name → entity_id map for resolving memory→entity edges
    let name_to_id: std::collections::HashMap<String, String> = entities
        .iter()
        .map(|e| (e.name.to_lowercase(), e.id.to_string()))
        .collect();

    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut graph_edges: Vec<GraphEdge> = Vec::new();

    // Memory nodes
    for m in &memories {
        let kind_str = format!("{:?}", m.kind).to_lowercase();
        nodes.push(GraphNode {
            id: m.id.to_string(),
            label: m.content.chars().take(60).collect::<String>(),
            node_type: "memory".to_string(),
            kind: kind_str,
            strength: m.strength,
            importance: m.importance,
            content: Some(m.content.clone()),
        });

        // Memory → entity edges via entity_refs
        for ref_name in &m.entity_refs {
            if let Some(entity_id) = name_to_id.get(&ref_name.to_lowercase()) {
                graph_edges.push(GraphEdge {
                    source: m.id.to_string(),
                    target: entity_id.clone(),
                    relation: "references".to_string(),
                });
            }
        }
    }

    // Entity nodes
    for e in &entities {
        let kind_str = format!("{:?}", e.kind).to_lowercase();
        nodes.push(GraphNode {
            id: e.id.to_string(),
            label: e.name.clone(),
            node_type: "entity".to_string(),
            kind: kind_str,
            strength: 1.0,
            importance: 0.5,
            content: None,
        });
    }

    // Entity → entity edges
    for edge in &edges {
        graph_edges.push(GraphEdge {
            source: edge.from_id.to_string(),
            target: edge.to_id.to_string(),
            relation: edge.relation.clone(),
        });
    }

    Json(GraphData {
        nodes,
        edges: graph_edges,
    })
    .into_response()
}
