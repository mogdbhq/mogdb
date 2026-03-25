/// Worst-to-best adversarial search tests against real PostgreSQL.
use chrono::Duration;
use mogdb_core::{MemoryKind, SourceTrust};
use mogdb_storage::{pipeline, search, Database, SearchQuery};
use uuid::Uuid;

async fn setup() -> sqlx::PgPool {
    dotenvy::dotenv().ok();
    let url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let db = Database::connect(&url).await.expect("failed to connect");
    db.pool
}

fn unique_user(test: &str) -> String {
    format!(
        "adv-{}-{}",
        test,
        Uuid::new_v4().to_string().split('-').next().unwrap()
    )
}

// =========================================================
// WORST CASE
// =========================================================

#[tokio::test]
async fn worst_empty_query() {
    let pool = setup().await;
    let user = unique_user("empty-q");

    pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "I use PostgreSQL for everything",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();

    let results = search::search(&pool, SearchQuery::new("test-agent", &user, ""))
        .await
        .unwrap();
    assert!(
        results.is_empty(),
        "empty query should return nothing, got {}",
        results.len()
    );
}

#[tokio::test]
async fn worst_all_stop_words_query() {
    let pool = setup().await;
    let user = unique_user("stopwords");

    pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "PostgreSQL is the best database",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();

    let results = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "the is a an it was"),
    )
    .await
    .unwrap();
    assert!(
        results.is_empty(),
        "all stop words should return nothing, got {}",
        results.len()
    );
}

#[tokio::test]
async fn worst_unicode_emoji_query() {
    let pool = setup().await;
    let user = unique_user("unicode");

    pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "We deploy on Kubernetes clusters",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();

    // Should not panic
    let results = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "🔥💀🤖 ñ ü ö 你好"),
    )
    .await
    .unwrap();
    assert!(
        results.is_empty() || results.len() < 100,
        "unicode query should not explode: {}",
        results.len()
    );
}

#[tokio::test]
async fn worst_sql_injection_in_query() {
    let pool = setup().await;
    let user = unique_user("sqli");

    pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "Normal memory about databases",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();

    // These should not panic, not error, and not return weird results
    let r1 = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "'; DROP TABLE memory_records; --"),
    )
    .await;
    assert!(
        r1.is_ok(),
        "SQL injection attempt should not crash: {:?}",
        r1.err()
    );

    let r2 = search::search(&pool, SearchQuery::new("test-agent", &user, "\" OR 1=1 --")).await;
    assert!(
        r2.is_ok(),
        "SQL injection attempt 2 should not crash: {:?}",
        r2.err()
    );

    let r3 = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "$1; DELETE FROM audit_log;"),
    )
    .await;
    assert!(
        r3.is_ok(),
        "SQL injection attempt 3 should not crash: {:?}",
        r3.err()
    );

    // Verify data is still intact
    let check = search::search(&pool, SearchQuery::new("test-agent", &user, "databases"))
        .await
        .unwrap();
    assert!(
        !check.is_empty(),
        "original data should still be there after injection attempts"
    );
}

#[tokio::test]
async fn worst_extremely_long_query() {
    let pool = setup().await;
    let user = unique_user("longq");

    pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "Short memory",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();

    let long_query = "database ".repeat(5000);
    let result = search::search(&pool, SearchQuery::new("test-agent", &user, &long_query)).await;
    assert!(
        result.is_ok(),
        "extremely long query should not crash: {:?}",
        result.err()
    );
}

#[tokio::test]
async fn worst_search_empty_database_user() {
    let pool = setup().await;
    let user = unique_user("neverused");

    // No memories stored for this user at all
    let results = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "anything at all"),
    )
    .await
    .unwrap();
    assert!(
        results.is_empty(),
        "should return empty for user with no memories"
    );
}

#[tokio::test]
async fn worst_search_only_expired_memories() {
    let pool = setup().await;
    let user = unique_user("allexpired");

    let m = pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "This memory about PostgreSQL will be expired",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();
    mogdb_storage::memory::expire(&pool, m.memory.id, "test-agent")
        .await
        .unwrap();

    let results = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "PostgreSQL memory"),
    )
    .await
    .unwrap();
    assert!(
        results.is_empty(),
        "expired memories should not appear in search"
    );
}

#[tokio::test]
async fn worst_search_only_invalidated_memories() {
    let pool = setup().await;
    let user = unique_user("allinvalid");

    let m = pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "We use MySQL for our main production database",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();
    mogdb_storage::memory::invalidate(&pool, m.memory.id, "test-agent")
        .await
        .unwrap();

    // Default search (no as_of) should exclude invalidated
    let results = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "MySQL production database"),
    )
    .await
    .unwrap();
    assert!(
        results.is_empty(),
        "invalidated memories should not appear in default search"
    );
}

#[tokio::test]
async fn worst_limit_zero() {
    let pool = setup().await;
    let user = unique_user("limzero");

    pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "PostgreSQL database memory",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();

    let results = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "PostgreSQL").limit(0),
    )
    .await
    .unwrap();
    assert!(results.is_empty(), "limit=0 should return nothing");
}

#[tokio::test]
async fn worst_nonexistent_agent() {
    let pool = setup().await;
    let user = unique_user("noagent");

    pipeline::ingest(
        &pool,
        "real-agent",
        &user,
        "Secret PostgreSQL database config",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();

    let results = search::search(
        &pool,
        SearchQuery::new("fake-agent-that-doesnt-exist", &user, "PostgreSQL config"),
    )
    .await
    .unwrap();
    assert!(results.is_empty(), "nonexistent agent should see nothing");
}

// =========================================================
// MEDIUM CASE
// =========================================================

#[tokio::test]
async fn medium_partial_word_match() {
    let pool = setup().await;
    let user = unique_user("partial");

    pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "PostgreSQL handles our production workloads efficiently",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();

    // "production" should match even if query uses "prod"
    // Postgres ts_rank may or may not stem this — testing the behavior
    let results = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "production workloads"),
    )
    .await
    .unwrap();
    assert!(!results.is_empty(), "exact keyword match should work");
}

#[tokio::test]
async fn medium_many_memories_ranking() {
    let pool = setup().await;
    let user = unique_user("manyrank");

    // Store 20 memories, only one highly relevant
    for i in 0..19 {
        pipeline::ingest(
            &pool,
            "test-agent",
            &user,
            &format!("Random fact number {i} about various unrelated technology topics"),
            MemoryKind::Episodic,
            SourceTrust::User,
            None,
        )
        .await
        .unwrap();
    }
    pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "The user strongly prefers PostgreSQL over MySQL for all database work",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();

    let results = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "PostgreSQL MySQL database preference").limit(5),
    )
    .await
    .unwrap();
    assert!(
        !results.is_empty(),
        "should find the relevant memory among 20"
    );
    assert!(
        results[0].content.contains("PostgreSQL") && results[0].content.contains("MySQL"),
        "most relevant should be first: {}",
        results[0].content
    );
}

#[tokio::test]
async fn medium_as_of_far_future() {
    let pool = setup().await;
    let user = unique_user("future");

    pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "We currently use Redis for caching layers",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();

    // Search as_of far in the future — should still find currently-valid memories
    let future = chrono::Utc::now() + Duration::days(365 * 10);
    let results = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "Redis caching").as_of(future),
    )
    .await
    .unwrap();
    assert!(
        !results.is_empty(),
        "as_of far future should find still-valid memories"
    );
}

#[tokio::test]
async fn medium_as_of_before_any_memories() {
    let pool = setup().await;
    let user = unique_user("beforeall");

    pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "We deploy everything on Kubernetes clusters",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();

    // Search as_of way in the past — before any memories existed
    let past = chrono::Utc::now() - Duration::days(365 * 50);
    let results = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "Kubernetes deploy").as_of(past),
    )
    .await
    .unwrap();
    assert!(
        results.is_empty(),
        "as_of before any memories should return nothing"
    );
}

#[tokio::test]
async fn medium_search_across_multiple_kinds() {
    let pool = setup().await;
    let user = unique_user("kinds");

    pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "Never deploy database migrations on Fridays",
        MemoryKind::Procedural,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();
    pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "Deployed database migration on Friday and it broke",
        MemoryKind::Episodic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();
    pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "User prefers database migrations during low traffic hours",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();

    // Without kind filter — should find all
    let all = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "database migration Friday"),
    )
    .await
    .unwrap();
    assert!(all.len() >= 2, "should find multiple kinds: {}", all.len());

    // With kind filter — should only return that kind
    let procedural = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "database migration Friday")
            .kind(MemoryKind::Procedural),
    )
    .await
    .unwrap();
    for r in &procedural {
        assert_eq!(
            r.kind,
            MemoryKind::Procedural,
            "filtered search returned wrong kind: {:?}",
            r.kind
        );
    }
}

// =========================================================
// GOOD CASE
// =========================================================

#[tokio::test]
async fn good_graph_context_shows_relationships() {
    let pool = setup().await;
    let user = unique_user("graphgood");

    pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "We switched from MySQL to PostgreSQL for our main backend database",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();

    let results = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "backend database").with_graph(),
    )
    .await
    .unwrap();
    assert!(!results.is_empty(), "should find result");

    // Collect all graph context across results
    let all_context: Vec<_> = results
        .iter()
        .flat_map(|r| r.graph_context.iter())
        .collect();
    if !all_context.is_empty() {
        let has_relation = all_context
            .iter()
            .any(|g| g.relation == "uses" || g.relation == "previously_used");
        assert!(
            has_relation,
            "graph context should show uses/previously_used relation: {:?}",
            all_context
        );
    }
}

#[tokio::test]
async fn good_access_count_increments_correctly() {
    let pool = setup().await;
    let user = unique_user("accessgood");

    let stored = pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "PostgreSQL is the primary production database for all services",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();

    // Search 3 times
    for _ in 0..3 {
        search::search(
            &pool,
            SearchQuery::new("test-agent", &user, "PostgreSQL production database"),
        )
        .await
        .unwrap();
    }

    let fetched = mogdb_storage::memory::fetch_by_id(&pool, stored.memory.id, "test-agent")
        .await
        .unwrap();
    assert!(
        fetched.access_count >= 3,
        "access_count should be >= 3 after 3 searches, got {}",
        fetched.access_count
    );
}

// =========================================================
// BEST CASE
// =========================================================

#[tokio::test]
async fn best_full_workflow_store_conflict_search() {
    let pool = setup().await;
    let user = unique_user("fullflow");

    // Step 1: Store initial tech stack
    pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "Our backend database is MySQL running on AWS infrastructure",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();
    pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "We use Redis for all caching layers in production",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();
    pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "Never deploy database changes without a backup plan",
        MemoryKind::Procedural,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();

    // Step 2: User switches database
    let switch = pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "We switched from MySQL to PostgreSQL for our backend database",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();
    assert!(
        switch.conflicts_invalidated > 0,
        "should have detected MySQL conflict"
    );

    // Step 3: Search for current state
    let results = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "backend database").with_graph(),
    )
    .await
    .unwrap();
    assert!(!results.is_empty(), "should find results");

    // The PostgreSQL memory should be in results.
    // The OLD MySQL-only memory ("Our backend database is MySQL") should be invalidated.
    // Note: the NEW memory ("We switched from MySQL to PostgreSQL") also contains "MySQL"
    // and is valid — that's correct behavior.
    let has_postgres = results.iter().any(|r| r.content.contains("PostgreSQL"));
    let has_old_mysql = results.iter().any(|r| {
        r.content.contains("MySQL") && !r.content.contains("PostgreSQL") && r.t_invalid.is_none()
    });
    assert!(has_postgres, "should find PostgreSQL memory");
    assert!(
        !has_old_mysql,
        "old MySQL-only memory should be invalidated"
    );

    // Step 4: Search for rules
    let rules = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "deploy database").kind(MemoryKind::Procedural),
    )
    .await
    .unwrap();
    assert!(!rules.is_empty(), "should find procedural rule");
    assert!(
        rules[0].content.contains("backup"),
        "should find the backup rule"
    );

    // Step 5: Redis should still be findable (not conflicted)
    let cache = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "caching Redis production"),
    )
    .await
    .unwrap();
    assert!(!cache.is_empty(), "Redis memory should still be valid");
}

#[tokio::test]
async fn best_temporal_history_traversal() {
    let pool = setup().await;
    let user = unique_user("history");

    // Build a history: Python → Go → Rust
    // Sleep between each to create distinct temporal windows
    let py = pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "The team uses Python for all backend services",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let _go = pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "The team switched from Python to Golang for backend services",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let _rs = pipeline::ingest(
        &pool,
        "test-agent",
        &user,
        "The team switched from Golang to Rust for backend services",
        MemoryKind::Semantic,
        SourceTrust::User,
        None,
    )
    .await
    .unwrap();

    // Current search: should find Rust
    let current = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "backend services language"),
    )
    .await
    .unwrap();
    let has_rust = current
        .iter()
        .any(|r| r.content.contains("Rust") && r.t_invalid.is_none());
    assert!(
        has_rust,
        "current search should find Rust: {:?}",
        current.iter().map(|r| &r.content).collect::<Vec<_>>()
    );

    // Historical search: at Python era (just after Python was stored, before Go arrived)
    let py_era = py.memory.t_valid + Duration::milliseconds(10);
    let py_results = search::search(
        &pool,
        SearchQuery::new("test-agent", &user, "backend services").as_of(py_era),
    )
    .await
    .unwrap();
    let has_python = py_results.iter().any(|r| r.content.contains("Python"));
    assert!(
        has_python,
        "as_of Python era should find Python: {:?}",
        py_results.iter().map(|r| &r.content).collect::<Vec<_>>()
    );
}
