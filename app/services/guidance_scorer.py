import logging
from typing import List, Dict
from collections import Counter

logger = logging.getLogger(__name__)

# Severity levels for queries (auto-detected from language patterns)
SEVERITY_PATTERNS = {
    "high": [
        r"\b(suicide|die|end it|hopeless|can't go on|worthless)\b",
        r"\b(severe|extreme|terrible|devastating|broken)\b",
    ],
    "medium": [
        r"\b(worried|anxious|stressed|struggling|hard|difficult)\b",
        r"\b(sad|grief|loss|pain|hurt|angry)\b",
    ],
    "low": [
        r"\b(curious|wonder|learn|understand|explore)\b",
        r"\b(general|sometimes|occasionally|mild)\b",
    ]
}

# Category diversity rules
CATEGORY_GROUPS = {
    "spiritual": ["Supplication & Spirituality (Dua, Dhikr, Tazkiyah)", "Faith (Aqeedah)"],
    "practical": ["Ethics & Morality (Akhlaq)", "Social Relations (Mu'amalat)", "Law (Ahkam)"],
    "narrative": ["History & Stories (Qasas al-Anbiya)", "Revelation"],
    "eschatological": ["Eschatology (Akhirah)", "Divine Attributes & Signs (Asma wa Sifat)"],
}

def detect_query_severity(query: str) -> str:
    """
    Auto-detect query severity from language patterns.
    No manual classification — purely pattern-based.
    """
    import re
    query_lower = query.lower()
    
    for severity, patterns in SEVERITY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                logger.debug(f"Query severity: {severity} (matched: {pattern})")
                return severity
    
    return "low"  # Default

def compute_severity_penalty(verse_emotions: List[str], verse_context: List[str], query_severity: str) -> float:
    """
    Penalize verses that are mismatched to query severity.
    
    Example: Don't show heavy "Warning" verses for mild queries.
    """
    penalty = 0.0
    
    heavy_emotions = ["Warning", "Fear", "Punishment", "Hell", "Anger at Injustice"]
    light_emotions = ["Comfort", "Hope", "Mercy", "Joy", "Gratitude"]
    
    if query_severity == "low":
        # Penalize heavy emotions for light queries
        if any(em in verse_emotions for em in heavy_emotions):
            penalty -= 0.15
            logger.debug(f"Penalty: heavy emotion for low-severity query")
    
    elif query_severity == "high":
        # Penalize light-only verses for severe queries (need depth)
        if all(em in light_emotions for em in verse_emotions) and len(verse_emotions) > 0:
            penalty -= 0.05
            logger.debug(f"Penalty: only light emotions for high-severity query")
    
    # Context mismatch penalties
    if query_severity in ["high", "medium"]:
        if "Law-giving context" in verse_context:
            penalty -= 0.1  # Legal verses feel cold for emotional distress
            logger.debug(f"Penalty: law-giving context for emotional query")
    
    return round(penalty, 4)

def enforce_diversity(candidates: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    Ensure category diversity in final results.
    Prevents echo chambers (e.g., 3 verses from same category).
    """
    if len(candidates) <= top_k:
        return candidates
    
    selected = []
    categories_used = Counter()
    group_counts = {group: 0 for group in CATEGORY_GROUPS.keys()}
    
    # Sort by score first
    sorted_candidates = sorted(candidates, key=lambda x: x["relevance_score"], reverse=True)
    
    for candidate in sorted_candidates:
        if len(selected) >= top_k:
            break
        
        category = candidate.get("category", "Unknown")
        
        # Find which group this category belongs to
        candidate_group = None
        for group, cats in CATEGORY_GROUPS.items():
            if category in cats:
                candidate_group = group
                break
        
        # Diversity rules:
        # 1. Max 2 verses from same category
        # 2. Max 2 verses from same group (for top_k=3)
        if categories_used[category] >= 2:
            logger.debug(f"Skip: category {category} already has 2 verses")
            continue
        
        if candidate_group and group_counts[candidate_group] >= 2 and len(selected) >= 2:
            logger.debug(f"Skip: group {candidate_group} already has 2 verses")
            continue
        
        selected.append(candidate)
        categories_used[category] += 1
        if candidate_group:
            group_counts[candidate_group] += 1
    
    # If we couldn't fill top_k with diversity rules, relax and add remaining
    if len(selected) < top_k:
        for candidate in sorted_candidates:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= top_k:
                break
    
    logger.info(f"Diversity enforcement: {len(candidates)} → {len(selected)} verses")
    return selected

def compute_repetition_penalty(verse_key: str, recent_verses: List[str]) -> float:
    """
    Penalize verses shown recently (requires external tracking).
    
    verse_key: "surah:ayah" format
    recent_verses: list of recently shown verse keys
    """
    if verse_key in recent_verses:
        # Strong penalty for exact repeat
        return -0.3
    
    # Partial penalty for same surah
    surah = verse_key.split(":")[0]
    same_surah_count = sum(1 for v in recent_verses if v.startswith(f"{surah}:"))
    if same_surah_count >= 2:
        return -0.1
    
    return 0.0
