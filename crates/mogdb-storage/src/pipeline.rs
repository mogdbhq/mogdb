/// The full write pipeline — orchestrates extraction, scoring, conflict
/// detection, entity graph updates, and storage into a single `ingest()` call.
use mogdb_core::{MemoryKind, MemoryRecord, MogError, NewMemoryRecord, SourceTrust};
use pgvector::Vector;
use sqlx::PgPool;
use tracing::{debug, info, warn};

use crate::{conflict, embedding::EmbeddingProvider, entity, extraction, memory, scoring};

/// Cosine similarity threshold for creating semantic links between memories.
const SEMANTIC_LINK_THRESHOLD: f64 = 0.7;

/// Result of ingesting a memory — includes what happened during the pipeline.
#[derive(Debug)]
pub struct IngestResult {
    /// The stored memory record.
    pub memory: MemoryRecord,
    /// How many existing memories were invalidated due to conflicts.
    pub conflicts_invalidated: u64,
    /// Entities that were created or updated.
    pub entities_touched: Vec<String>,
    /// Whether the memory was quarantined (external source).
    pub quarantined: bool,
}

/// Ingest raw text into MogDB's memory system.
///
/// This is the main entry point for the write path. It:
/// 1. Extracts entities from the text
/// 2. Scores importance
/// 3. Detects and invalidates conflicting memories
/// 4. Stores the memory record
/// 5. Updates the entity graph
pub async fn ingest(
    pool: &PgPool,
    agent_id: &str,
    user_id: &str,
    content: &str,
    kind: MemoryKind,
    source_trust: SourceTrust,
    session_id: Option<&str>,
) -> Result<IngestResult, MogError> {
    // Step 1: Extract entities
    let extracted = extraction::extract_entities(content);
    let entity_names: Vec<String> = extracted.iter().map(|e| e.name.clone()).collect();

    debug!(
        entities = ?entity_names,
        "extracted {} entities from content",
        extracted.len()
    );

    // Step 2: Score importance
    let is_procedural = kind == MemoryKind::Procedural;
    let importance = scoring::score_importance(content, is_procedural);

    debug!(importance, "scored importance");

    // Step 3: Check if quarantined
    let quarantined = source_trust == SourceTrust::External;
    if quarantined {
        warn!("memory from external source — will be quarantined");
    }

    // Step 4: Detect conflicts (skip for quarantined memories)
    // We defer invalidation until after the new memory is stored so we can
    // create supersedes edges (version chains).
    let mut conflicts_invalidated = 0u64;
    let mut detected_conflict_ids: Vec<uuid::Uuid> = Vec::new();
    if !quarantined {
        let candidates =
            conflict::find_conflicts(pool, agent_id, user_id, content, &entity_names).await?;

        if !candidates.is_empty() {
            debug!("found {} conflict candidates", candidates.len());

            detected_conflict_ids = candidates
                .iter()
                .filter(|c| conflict::is_contradicting(&c.content, content, &entity_names))
                .map(|c| {
                    info!(
                        id = %c.id,
                        old_content = c.content,
                        "will invalidate conflicting memory"
                    );
                    c.id
                })
                .collect();
        }
    }

    // Step 5: Store the memory record
    let input = NewMemoryRecord {
        agent_id: agent_id.to_string(),
        user_id: user_id.to_string(),
        session_id: session_id.map(|s| s.to_string()),
        content: content.to_string(),
        kind,
        source_trust,
        t_valid: None,
        importance: Some(importance),
        entity_refs: entity_names.clone(),
        provenance: None,
    };

    let record = memory::store(pool, input).await?;

    // Step 5b: Invalidate conflicts with version chain (supersedes edges)
    if !detected_conflict_ids.is_empty() {
        conflicts_invalidated = conflict::invalidate_conflicts_with_chain(
            pool,
            agent_id,
            &detected_conflict_ids,
            Some(record.id),
        )
        .await?;
    }

    // Step 6: Update entity graph (skip for quarantined memories)
    let mut entities_touched = Vec::new();
    if !quarantined {
        // Create/update entity nodes
        for ext in &extracted {
            let _entity_id = entity::upsert(pool, agent_id, user_id, &ext.name, &ext.kind).await?;
            entities_touched.push(ext.name.clone());
        }

        // Extract and create relationship edges
        let relations = extraction::extract_relations(content);
        for (subject_hint, relation, object_hint) in &relations {
            // Find or create subject entity (use _user as a stand-in for the user)
            let subj_name = if subject_hint == "_user" {
                user_id
            } else {
                subject_hint
            };

            // Try to find the object entity among our extracted entities
            let obj_entity = extracted
                .iter()
                .find(|e| object_hint.to_lowercase().contains(&e.name.to_lowercase()));

            if let Some(obj) = obj_entity {
                let subj_id = entity::upsert(
                    pool,
                    agent_id,
                    user_id,
                    subj_name,
                    &mogdb_core::EntityKind::Person,
                )
                .await?;
                let obj_id = entity::upsert(pool, agent_id, user_id, &obj.name, &obj.kind).await?;

                // If this is a "previously_used" relation, invalidate existing "uses" edges
                if relation == "previously_used" || relation == "stopped_using" {
                    entity::invalidate_edges(pool, subj_id, obj_id, "uses").await?;
                }

                entity::create_edge(
                    pool,
                    subj_id,
                    obj_id,
                    relation,
                    Some(record.id),
                    Some(record.t_valid),
                )
                .await?;

                debug!(
                    subject = subj_name,
                    relation,
                    object = obj.name,
                    "created entity edge"
                );
            }
        }
    }

    info!(
        id = %record.id,
        importance,
        conflicts_invalidated,
        entities = entities_touched.len(),
        quarantined,
        "memory ingested (basic)"
    );

    Ok(IngestResult {
        memory: record,
        conflicts_invalidated,
        entities_touched,
        quarantined,
    })
}

/// Same as `ingest()` but also:
/// 1. Optionally decomposes content into atomic facts (MOGDB_DECOMPOSE=1)
/// 2. Runs LLM-based entity extraction (catches entities keywords miss)
/// 3. Generates and stores an embedding vector
///
/// When fact decomposition is enabled, the original memory is stored as a
/// "parent" and then decomposed into atomic fact memories. Each fact gets
/// its own entity extraction, embedding, and importance score. The parent
/// memory is soft-expired since the facts replace it for retrieval.
#[allow(clippy::too_many_arguments)]
pub async fn ingest_with_embedder<E: EmbeddingProvider>(
    pool: &PgPool,
    agent_id: &str,
    user_id: &str,
    content: &str,
    kind: MemoryKind,
    source_trust: SourceTrust,
    session_id: Option<&str>,
    embedder: &E,
) -> Result<IngestResult, MogError> {
    // Step 1: Atomic fact decomposition (if enabled)
    if extraction::decompose_enabled() {
        let facts = extraction::decompose_facts(content).await;

        // If decomposition produced multiple facts, store each independently
        if facts.len() > 1 {
            debug!(
                facts = facts.len(),
                "decomposed into atomic facts, ingesting each"
            );

            // Store the original as parent (for provenance), then expire it
            let parent = ingest(
                pool,
                agent_id,
                user_id,
                content,
                kind.clone(),
                source_trust.clone(),
                session_id,
            )
            .await?;

            // Expire the parent — the atomic facts replace it for retrieval
            let _ = memory::expire(pool, parent.memory.id, agent_id).await;

            // Ingest each atomic fact with full pipeline
            let mut total_entities = parent.entities_touched.clone();
            let mut total_conflicts = parent.conflicts_invalidated;
            let mut first_fact_record = None;

            for fact in &facts {
                match ingest_single_with_embedder(
                    pool,
                    agent_id,
                    user_id,
                    fact,
                    kind.clone(),
                    source_trust.clone(),
                    session_id,
                    embedder,
                    Some(parent.memory.id),
                )
                .await
                {
                    Ok(fact_result) => {
                        total_conflicts += fact_result.conflicts_invalidated;
                        for ent in &fact_result.entities_touched {
                            if !total_entities.iter().any(|e| e.eq_ignore_ascii_case(ent)) {
                                total_entities.push(ent.clone());
                            }
                        }
                        if first_fact_record.is_none() {
                            first_fact_record = Some(fact_result.memory);
                        }
                    }
                    Err(e) => {
                        warn!(error = %e, "failed to ingest atomic fact");
                    }
                }
            }

            // Return the first fact as the "result" memory (for API compatibility)
            return Ok(IngestResult {
                memory: first_fact_record.unwrap_or(parent.memory),
                conflicts_invalidated: total_conflicts,
                entities_touched: total_entities,
                quarantined: parent.quarantined,
            });
        }
    }

    // Step 2: Normal single-memory path (no decomposition or single fact)
    ingest_single_with_embedder(
        pool,
        agent_id,
        user_id,
        content,
        kind,
        source_trust,
        session_id,
        embedder,
        None,
    )
    .await
}

/// Ingest a single memory with LLM entity extraction and embedding.
#[allow(clippy::too_many_arguments)]
async fn ingest_single_with_embedder<E: EmbeddingProvider>(
    pool: &PgPool,
    agent_id: &str,
    user_id: &str,
    content: &str,
    kind: MemoryKind,
    source_trust: SourceTrust,
    session_id: Option<&str>,
    embedder: &E,
    provenance: Option<uuid::Uuid>,
) -> Result<IngestResult, MogError> {
    // Store the memory (with optional provenance link to parent)
    let extracted = extraction::extract_entities(content);
    let entity_names: Vec<String> = extracted.iter().map(|e| e.name.clone()).collect();
    let is_procedural = kind == MemoryKind::Procedural;
    let importance = scoring::score_importance(content, is_procedural);
    let quarantined = source_trust == SourceTrust::External;

    // Conflict detection — with version chain creation (Task 6)
    let mut conflicts_invalidated = 0u64;
    // We'll store conflict_ids so we can create supersedes edges after the record is stored
    let mut detected_conflict_ids: Vec<uuid::Uuid> = Vec::new();
    if !quarantined {
        let candidates =
            conflict::find_conflicts(pool, agent_id, user_id, content, &entity_names).await?;
        if !candidates.is_empty() {
            detected_conflict_ids = candidates
                .iter()
                .filter(|c| conflict::is_contradicting(&c.content, content, &entity_names))
                .map(|c| c.id)
                .collect();
        }
    }

    let input = NewMemoryRecord {
        agent_id: agent_id.to_string(),
        user_id: user_id.to_string(),
        session_id: session_id.map(|s| s.to_string()),
        content: content.to_string(),
        kind,
        source_trust,
        t_valid: None,
        importance: Some(importance),
        entity_refs: entity_names.clone(),
        provenance,
    };

    let record = memory::store(pool, input).await?;

    // Now invalidate conflicts with version chain (supersedes edges)
    if !detected_conflict_ids.is_empty() {
        conflicts_invalidated = conflict::invalidate_conflicts_with_chain(
            pool,
            agent_id,
            &detected_conflict_ids,
            Some(record.id),
        )
        .await?;
    }

    // Entity graph
    let mut entities_touched = Vec::new();
    if !quarantined {
        for ext in &extracted {
            let _ = entity::upsert(pool, agent_id, user_id, &ext.name, &ext.kind).await?;
            entities_touched.push(ext.name.clone());
        }

        let relations = extraction::extract_relations(content);
        for (subject_hint, relation, object_hint) in &relations {
            let subj_name = if subject_hint == "_user" {
                user_id
            } else {
                subject_hint
            };

            let obj_entity = extracted
                .iter()
                .find(|e| object_hint.to_lowercase().contains(&e.name.to_lowercase()));

            if let Some(obj) = obj_entity {
                let subj_id = entity::upsert(
                    pool,
                    agent_id,
                    user_id,
                    subj_name,
                    &mogdb_core::EntityKind::Person,
                )
                .await?;
                let obj_id = entity::upsert(pool, agent_id, user_id, &obj.name, &obj.kind).await?;

                if relation == "previously_used" || relation == "stopped_using" {
                    entity::invalidate_edges(pool, subj_id, obj_id, "uses").await?;
                }

                entity::create_edge(
                    pool,
                    subj_id,
                    obj_id,
                    relation,
                    Some(record.id),
                    Some(record.t_valid),
                )
                .await?;
            }
        }
    }

    // LLM entity extraction + embedding (parallel, skip for quarantined)
    if !quarantined {
        let llm_future = extraction::extract_entities_with_llm(content);
        let embed_future = embedder.embed(content);
        let (llm_result, embed_result) = tokio::join!(llm_future, embed_future);

        // Process LLM entities
        let (llm_entities, llm_relations) = llm_result;
        for ent in &llm_entities {
            if !entities_touched
                .iter()
                .any(|n| n.eq_ignore_ascii_case(&ent.name))
            {
                if let Ok(_id) = entity::upsert(pool, agent_id, user_id, &ent.name, &ent.kind).await
                {
                    entities_touched.push(ent.name.clone());
                }
            }
        }

        if entities_touched.len() > entity_names.len() {
            let _ = memory::update_entity_refs(pool, record.id, &entities_touched).await;
        }

        // Process LLM relations
        for (subject_hint, relation, object_hint) in &llm_relations {
            let subj_name = if subject_hint == "_user"
                || subject_hint.eq_ignore_ascii_case("user")
                || subject_hint.eq_ignore_ascii_case("I")
            {
                user_id
            } else {
                subject_hint
            };
            let obj_entity = llm_entities
                .iter()
                .find(|e| object_hint.to_lowercase().contains(&e.name.to_lowercase()));
            if let Some(obj) = obj_entity {
                if let (Ok(subj_id), Ok(obj_id)) = (
                    entity::upsert(
                        pool,
                        agent_id,
                        user_id,
                        subj_name,
                        &mogdb_core::EntityKind::Person,
                    )
                    .await,
                    entity::upsert(pool, agent_id, user_id, &obj.name, &obj.kind).await,
                ) {
                    if relation == "previously_used" || relation == "stopped_using" {
                        let _ = entity::invalidate_edges(pool, subj_id, obj_id, "uses").await;
                    }
                    let _ = entity::create_edge(
                        pool,
                        subj_id,
                        obj_id,
                        relation,
                        Some(record.id),
                        Some(record.t_valid),
                    )
                    .await;
                }
            }
        }

        // Embedding + semantic links (Task 4)
        match embed_result {
            Ok(vec) => {
                memory::store_embedding(pool, record.id, vec.clone()).await?;

                // Create semantic links to similar recent memories
                let _ = create_semantic_links(pool, agent_id, user_id, record.id, &vec).await;
            }
            Err(e) => {
                warn!(id = %record.id, error = %e, "embedding failed — stored without vector");
            }
        }
    }

    info!(
        id = %record.id,
        importance,
        conflicts_invalidated,
        entities = entities_touched.len(),
        quarantined,
        "memory ingested (with_embedder)"
    );

    Ok(IngestResult {
        memory: record,
        conflicts_invalidated,
        entities_touched,
        quarantined,
    })
}

/// Create semantic links between the new memory and existing similar memories.
///
/// Finds recent memories with cosine similarity > SEMANTIC_LINK_THRESHOLD and
/// creates `semantically_related` edges with similarity as weight.
/// This connects memories across sessions even when they share no entities.
async fn create_semantic_links(
    pool: &PgPool,
    agent_id: &str,
    user_id: &str,
    memory_id: uuid::Uuid,
    embedding: &[f32],
) -> Result<u64, MogError> {
    let vec = Vector::from(embedding.to_vec());

    // Find similar memories (cosine distance < 1 - threshold)
    let max_distance = 1.0 - SEMANTIC_LINK_THRESHOLD;
    let similar = sqlx::query_as::<_, (uuid::Uuid, f64)>(
        r#"
        SELECT id, (embedding <=> $3) AS distance
        FROM memory_records
        WHERE agent_id = $1
          AND user_id = $2
          AND id != $4
          AND t_expired IS NULL
          AND quarantined = false
          AND embedding IS NOT NULL
          AND (embedding <=> $3) < $5
        ORDER BY distance ASC
        LIMIT 5
        "#,
    )
    .bind(agent_id)
    .bind(user_id)
    .bind(&vec)
    .bind(memory_id)
    .bind(max_distance)
    .fetch_all(pool)
    .await?;

    let mut links_created = 0u64;
    for (similar_id, distance) in &similar {
        let similarity = 1.0 - distance;
        let edge_id = uuid::Uuid::new_v4();
        let result = sqlx::query(
            r#"
            INSERT INTO entity_edges (id, from_id, to_id, relation, weight, t_valid, t_invalid, source_memory)
            VALUES ($1, $2, $3, 'semantically_related', $4, NOW(), NULL, $5)
            "#,
        )
        .bind(edge_id)
        .bind(memory_id)
        .bind(similar_id)
        .bind(similarity)
        .bind(memory_id)
        .execute(pool)
        .await;

        if result.is_ok() {
            links_created += 1;
            debug!(
                from = %memory_id,
                to = %similar_id,
                similarity,
                "created semantic link"
            );
        }
    }

    if links_created > 0 {
        info!(
            links_created,
            "semantic links created for memory {memory_id}"
        );
    }

    Ok(links_created)
}
