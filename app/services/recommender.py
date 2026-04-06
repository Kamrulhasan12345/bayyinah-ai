import logging
import json
import os
from app.services.guidance_scorer import compute_repetition_penalty, compute_severity_penalty, detect_query_severity, enforce_diversity
from app.services.loader import load_dataset
from app.services.embeddings import semantic_search, get_model, MODEL_NAME

logger = logging.getLogger(__name__)

INTENT_MAP_PATH = os.getenv("INTENT_MAP_PATH", "data/intent_map.json")
_intent_map = None

def load_intent_map() -> dict:
    global _intent_map

    if _intent_map is None:
        if os.path.exists(INTENT_MAP_PATH):
            logger.info(f"Loading intent map from {INTENT_MAP_PATH}")
            with open(INTENT_MAP_PATH, "r", encoding="utf-8") as f:
                _intent_map = json.load(f)
            logger.info("Intent map loaded successfully.")
        else:
            logger.warning(f"Intent map not found at {INTENT_MAP_PATH}. Running without intent clustering.")
            _intent_map = {"clusters": {}, "example_queries": {}, "metadata": {}}
    
    return _intent_map

def compute_metadata_boost(verse_tags: list, verse_emotions: list, verse_context: list, query: str) -> float:
    """
    Compute a metadata-based score boost based on tag/query alignment.
    
    This is where guidance logic lives - NOT in the embedding.
    """
    boost = 0.0
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    # Boost 1: Tag-query keyword overlap (lightweight semantic bridge)
    for tag in verse_tags:
        tag_lower = tag.lower()
        if any(word in tag_lower for word in query_words if len(word) > 3):
            boost += 0.05
    
    # Boost 2: Intent cluster alignment (if query matches a cluster's example patterns)
    intent_map = load_intent_map()
    for cluster_name, cluster_data in intent_map.get("clusters", {}).items():
        cluster_tags = [t.lower() for t in cluster_data.get("all_tags", [])]
        if any(tag_lower in cluster_tags for tag_lower in [t.lower() for t in verse_tags]):
            # Check if query semantically relates to this cluster
            if any(word in query_lower for word in cluster_name.lower().split()):
                boost += 0.1
    
    # Boost 3: Context appropriateness (prefer "Spiritual reminder" for emotional queries)
    emotional_keywords = ["sad", "anxious", "worried", "scared", "grief", "lost", "confused"]
    if any(kw in query_lower for kw in emotional_keywords):
        if "Spiritual reminder" in verse_context or "Daily life context" in verse_context:
            boost += 0.05
        # Penalize heavy legal contexts for emotional queries
        if "Law-giving context" in verse_context:
            boost -= 0.1
    
    return round(boost, 4)

def recommend_verses(query: str, top_k: int = 3, language: str = "english") -> dict:
    df = load_dataset()
    logger.info(f"Running recommendation for query: '{query}'")

    search_results = semantic_search(query, df, top_k=50)
    logger.info(f"Semantic search returned {len(search_results)} candidates. Applying metadata boosts...")

    ranked_results = []

    for df_idx, distance in search_results:
        verse = df.iloc[df_idx]

        semantic_score = 1 / (1 + distance)

        metadata_boost = compute_metadata_boost(
            verse_tags=verse['tags'],
            verse_emotions=verse['emotion'],
            verse_context=verse['context'],
            query=query
        )

        final_score = round(0.85 * semantic_score + 0.15 * min(1.0, metadata_boost), 4)

        if language == 'arabic':
            text = verse['arabic']
        elif language == 'urdu':
            text = verse['urdu']
        else:
            text = verse['english']

            ranked_results.append({
                'surah': int(verse['surah']),
                'ayah': int(verse['ayah']),
                'text': text,
                'arabic': verse['arabic'],
                'translation_en': verse['english'],
                'translation_ur': verse['urdu'],
                'emotion': verse['emotion'],
                'tags': verse['tags'],
                'category': verse['category'],
                'context': verse['context'],
                'relevance_score': final_score,
                'semantic_distance': round(distance, 4),
                'semantic_score': round(semantic_score, 4),
                'metadata_boost': metadata_boost
            })

    query_severity = detect_query_severity(query)
    logger.info(f"Detected query severity: {query_severity}. Applying severity penalties...")

    for result in ranked_results:
        severity_penalty = compute_severity_penalty(
            query_severity=query_severity,
            verse_emotions=result['emotion'],
            verse_context=result['context']
        )

        verse_key = f"{result['surah']}:{result['ayah']}"
        repetition_penalty = compute_repetition_penalty(verse_key, recent_verses=[])

        result['relevance_score'] = round(result['relevance_score'] + severity_penalty + repetition_penalty, 4)
        result['severity_penalty'] = severity_penalty
        result['repetition_penalty'] = repetition_penalty
    
    ranked_results.sort(key=lambda x: x['relevance_score'], reverse=True)
    final_results = enforce_diversity(ranked_results, top_k=top_k)


    logger.info(f"Returning top {len(final_results)} results for query: '{query}'")

    return {
        'query': query,
        'detected_emotions': [],
        'verses': final_results,
        'metadata': {
            'model': MODEL_NAME,
            'ranking': "hybrid (0.85 semantic + 0.15 metadata)",
            'language': language,
            'candidates_retrieved': len(search_results),
        }
    }
