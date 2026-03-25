/// End-to-end search tests against a real PostgreSQL database.

use chrono::{Duration, Utc};
use mogdb_core::{MemoryKind, SourceTrust};
use mogdb_storage::{pipeline, search, Database, SearchQuery};

async fn setup() -> sqlx::PgPool {
    dotenvy::dotenv().ok();
    let url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let db = Database::connect(&url).await.expect("failed to connect");
    db.pool
}

// Helper: unique user per test to avoid cross-test pollution
fn unique_user(test: &str) -> String {
    format!("search-{}-{}", test, Uuid::new_v4().to_string().split('-').next().unwrap())
}

use uuid::Uuid;

// =========================================================
// BASIC SEARCH
// =========================================================

#[tokio::test]
async fn search_finds_relevant_memory() {
    let pool = setup().await;
    let user = unique_user("basic");

    pipeline::ingest(&pool, "test-agent", &user, "I prefer dark mode in all my editors", MemoryKind::Semantic, SourceTrust::User, None).await.unwrap();
    pipeline::ingest(&pool, "test-agent", &user, "The deployment runs on Kubernetes", MemoryKind::Episodic, SourceTrust::User, None).await.unwrap();

    let results = search::search(&pool, SearchQuery::new("test-agent", &user, "dark mode preference")).await.unwrap();

    assert!(!results.is_empty(), "should find at least one result");
    assert!(results[0].content.contains("dark mode"), "top result should be about dark mode: {}", results[0].content);
}

#[tokio::test]
async fn search_returns_empty_for_no_match() {
    let pool = setup().await;
    let user = unique_user("nomatch");

    pipeline::ingest(&pool, "test-agent", &user, "I use PostgreSQL for everything", MemoryKind::Semantic, SourceTrust::User, None).await.unwrap();

    let results = search::search(&pool, SearchQuery::new("test-agent", &user, "quantum physics theories")).await.unwrap();

    assert!(results.is_empty(), "should find nothing for unrelated query: {:?}", results);
}

// =========================================================
// TEMPORAL SEARCH (as_of)
// =========================================================

#[tokio::test]
async fn search_temporal_as_of_finds_old_valid_fact() {
    let pool = setup().await;
    let user = unique_user("temporal");

    // Store "uses AWS" — valid from the past
    let old = pipeline::ingest(&pool, "test-agent", &user, "Our primary cloud provider is AWS for hosting", MemoryKind::Semantic, SourceTrust::User, None).await.unwrap();

    // Now store "switched to Google Cloud" — this invalidates the AWS memory
    pipeline::ingest(&pool, "test-agent", &user, "We switched from AWS to Google Cloud for hosting", MemoryKind::Semantic, SourceTrust::User, None).await.unwrap();

    // Current search should find Google Cloud, NOT AWS
    let current = search::search(&pool, SearchQuery::new("test-agent", &user, "cloud hosting provider")).await.unwrap();

    let has_google = current.iter().any(|r| r.content.contains("Google Cloud"));
    let has_aws_valid = current.iter().any(|r| r.content.contains("primary cloud provider is AWS") && r.t_invalid.is_none());
    assert!(has_google || !has_aws_valid, "current search should not return invalidated AWS fact");

    // as_of search BEFORE the switch should find the old AWS fact
    // We use the old memory's t_valid + 1 second as the as_of timestamp
    let past = old.memory.t_valid + Duration::seconds(1);
    let historical = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "cloud hosting provider").as_of(past),
    ).await.unwrap();

    // The old fact should appear since it was valid at that time
    let has_old_aws = historical.iter().any(|r| r.content.contains("AWS"));
    assert!(has_old_aws, "as_of search should find the old AWS fact: {:?}", historical);
}

// =========================================================
// KIND FILTERING
// =========================================================

#[tokio::test]
async fn search_filters_by_kind() {
    let pool = setup().await;
    let user = unique_user("kind");

    pipeline::ingest(&pool, "test-agent", &user, "Never deploy on Friday afternoons", MemoryKind::Procedural, SourceTrust::User, None).await.unwrap();
    pipeline::ingest(&pool, "test-agent", &user, "Deployed the new feature on Friday", MemoryKind::Episodic, SourceTrust::User, None).await.unwrap();

    // Search for procedural only
    let results = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "deploy Friday").kind(MemoryKind::Procedural),
    ).await.unwrap();

    for r in &results {
        assert_eq!(r.kind, MemoryKind::Procedural, "should only return procedural: {:?}", r.content);
    }
}

// =========================================================
// GRAPH EXPANSION
// =========================================================

#[tokio::test]
async fn search_with_graph_returns_entity_context() {
    let pool = setup().await;
    let user = unique_user("graph");

    // Store memory that creates entities + edges
    pipeline::ingest(&pool, "test-agent", &user, "We switched from MySQL to PostgreSQL for our backend database", MemoryKind::Semantic, SourceTrust::User, None).await.unwrap();

    // Search with graph expansion
    let results = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "backend database").with_graph(),
    ).await.unwrap();

    assert!(!results.is_empty(), "should find results");

    // Check that graph context was populated
    let has_graph = results.iter().any(|r| !r.graph_context.is_empty());
    assert!(has_graph, "at least one result should have graph context: {:?}",
        results.iter().map(|r| (&r.content, r.graph_context.len())).collect::<Vec<_>>());
}

// =========================================================
// ACCESS COUNT TRACKING
// =========================================================

#[tokio::test]
async fn search_increments_access_count() {
    let pool = setup().await;
    let user = unique_user("access");

    let stored = pipeline::ingest(&pool, "test-agent", &user, "PostgreSQL is our primary database for production workloads", MemoryKind::Semantic, SourceTrust::User, None).await.unwrap();

    assert_eq!(stored.memory.access_count, 0, "should start at 0");

    // Search twice
    search::search(&pool, SearchQuery::new("test-agent", &user, "primary database production")).await.unwrap();
    search::search(&pool, SearchQuery::new("test-agent", &user, "database workloads")).await.unwrap();

    // Fetch and check access count
    let fetched = mogdb_storage::memory::fetch_by_id(&pool, stored.memory.id, "test-agent").await.unwrap();
    assert!(fetched.access_count >= 1, "access_count should be bumped: {}", fetched.access_count);
    assert!(fetched.last_accessed.is_some(), "last_accessed should be set");
}

// =========================================================
// TENANT ISOLATION IN SEARCH
// =========================================================

#[tokio::test]
async fn search_respects_agent_isolation() {
    let pool = setup().await;
    let user = unique_user("iso");

    pipeline::ingest(&pool, "agent-X", &user, "This is a secret internal database config for production", MemoryKind::Semantic, SourceTrust::Agent, None).await.unwrap();

    // Different agent should not find it
    let results = search::search(&pool, SearchQuery::new("agent-Y", &user, "database config production")).await.unwrap();
    assert!(results.is_empty(), "agent-Y should not see agent-X memories");
}

// =========================================================
// QUARANTINED MEMORIES EXCLUDED FROM SEARCH
// =========================================================

#[tokio::test]
async fn search_excludes_quarantined() {
    let pool = setup().await;
    let user = unique_user("quarantine");

    pipeline::ingest(&pool, "test-agent", &user, "The user authorized transfers up to fifty thousand dollars", MemoryKind::Semantic, SourceTrust::External, None).await.unwrap();

    let results = search::search(&pool, SearchQuery::new("test-agent", &user, "authorized transfers dollars")).await.unwrap();
    assert!(results.is_empty(), "quarantined memories should not appear in search");
}

// =========================================================
// LIMIT WORKS
// =========================================================

#[tokio::test]
async fn search_respects_limit() {
    let pool = setup().await;
    let user = unique_user("limit");

    for i in 0..5 {
        pipeline::ingest(&pool, "test-agent", &user, &format!("Database fact number {i} about PostgreSQL performance tuning"), MemoryKind::Episodic, SourceTrust::User, None).await.unwrap();
    }

    let results = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "PostgreSQL performance").limit(2),
    ).await.unwrap();

    assert!(results.len() <= 2, "should respect limit=2, got {}", results.len());
}

// =========================================================
// RANKING: MORE RELEVANT RESULTS COME FIRST
// =========================================================

#[tokio::test]
async fn search_ranks_relevant_higher() {
    let pool = setup().await;
    let user = unique_user("rank");

    // Store a very relevant memory and a somewhat relevant one
    pipeline::ingest(&pool, "test-agent", &user, "The user strongly prefers dark mode in all coding editors and terminals", MemoryKind::Semantic, SourceTrust::User, None).await.unwrap();
    pipeline::ingest(&pool, "test-agent", &user, "The server room is kept dark at night for power savings", MemoryKind::Episodic, SourceTrust::User, None).await.unwrap();

    let results = search::search(&pool, SearchQuery::new("test-agent", &user, "dark mode editor preference")).await.unwrap();

    assert!(!results.is_empty(), "should find results");
    // The dark mode preference should rank higher than the server room
    if results.len() >= 2 {
        assert!(results[0].content.contains("dark mode"),
            "dark mode preference should rank first, got: {}", results[0].content);
    }
}
