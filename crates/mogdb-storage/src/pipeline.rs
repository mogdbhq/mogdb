/// The full write pipeline — orchestrates extraction, scoring, conflict
/// detection, entity graph updates, and storage into a single `ingest()` call.
use mogdb_core::{MemoryKind, MemoryRecord, MogError, NewMemoryRecord, SourceTrust};
use sqlx::PgPool;
use tracing::{debug, info, warn};

use crate::{conflict, embedding::EmbeddingProvider, entity, extraction, memory, scoring};

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

    // Step 4: Detect and invalidate conflicts (skip for quarantined memories)
    let mut conflicts_invalidated = 0u64;
    if !quarantined {
        let candidates =
            conflict::find_conflicts(pool, agent_id, user_id, content, &entity_names).await?;

        if !candidates.is_empty() {
            debug!("found {} conflict candidates", candidates.len());

            let conflict_ids: Vec<uuid::Uuid> = candidates
                .iter()
                .filter(|c| conflict::is_contradicting(&c.content, content, &entity_names))
                .map(|c| {
                    info!(
                        id = %c.id,
                        old_content = c.content,
                        "invalidating conflicting memory"
                    );
                    c.id
                })
                .collect();

            conflicts_invalidated =
                conflict::invalidate_conflicts(pool, agent_id, &conflict_ids).await?;
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

                entity::create_edge(pool, subj_id, obj_id, relation, Some(record.id)).await?;

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
        "memory ingested"
    );

    Ok(IngestResult {
        memory: record,
        conflicts_invalidated,
        entities_touched,
        quarantined,
    })
}

/// Same as `ingest()` but also generates and stores an embedding vector.
/// The embedding is generated after the record is stored, so a storage failure
/// never leaves an orphaned vector.
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
    let result = ingest(
        pool,
        agent_id,
        user_id,
        content,
        kind,
        source_trust,
        session_id,
    )
    .await?;

    // Skip embedding for quarantined memories — they're unreviewed external content.
    if !result.quarantined {
        match embedder.embed(content).await {
            Ok(vec) => {
                memory::store_embedding(pool, result.memory.id, vec).await?;
                debug!(id = %result.memory.id, "stored embedding");
            }
            Err(e) => {
                // Embedding failure is non-fatal: the memory is stored, just without a vector.
                warn!(id = %result.memory.id, error = %e, "embedding generation failed — memory stored without vector");
            }
        }
    }

    Ok(result)
}
