import os
import json
import hashlib
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Cache for LLM responses (in-memory for MVP, use Redis in prod)
_reflection_cache: Dict[str, Dict] = {}

# Prompt template for reflections
REFLECTION_PROMPT = """You are a compassionate Islamic guide. Your role is to help users connect Quranic verses to their personal situation with warmth and wisdom.

USER'S SITUATION: {query}

QURANIC VERSE: Surah {surah}: Ayah {ayah}
Arabic: {arabic}
Translation: {translation}

VERSE THEMES: {themes}

GUIDELINES:
1. Write 2-3 sentences maximum
2. Connect the verse to the user's situation gently
3. Do NOT give religious rulings (fatwa)
4. Do NOT claim to speak for Allah or scholars
5. Use warm, supportive language
6. End with a brief reflection prompt or du'a suggestion
7. If the query involves severe distress, gently suggest seeking human support

REFLECTION:"""

def _get_cache_key(query: str, surah: int, ayah: int) -> str:
    """Generate unique cache key for query + verse combination"""
    content = f"{query}|{surah}:{ayah}"
    return hashlib.md5(content.encode()).hexdigest()

def _is_cache_valid(cache_entry: Dict) -> bool:
    """Check if cached response is still valid (1 hour TTL)"""
    if not cache_entry:
        return False
    cached_at = datetime.fromisoformat(cache_entry["cached_at"])
    return datetime.now() - cached_at < timedelta(hours=1)

def generate_reflection_groq(query: str, verse: Dict) -> Optional[str]:
    """Generate reflection using Groq (free tier, fast)"""
    try:
        from groq import Groq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not set, skipping Groq")
            return None
        
        client = Groq(api_key=api_key)
        
        prompt = REFLECTION_PROMPT.format(
            query=query[:200],  # Truncate long queries
            surah=verse["surah"],
            ayah=verse["ayah"],
            arabic=verse["arabic"][:100],
            translation=verse["translation_en"][:300],
            themes=", ".join(verse.get("tags", [])[:5])
        )
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Free, fast model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Low temperature for consistency
            max_tokens=200
        )
        
        reflection = response.choices[0].message.content
        if reflection:
            reflection = reflection.strip()
        logger.info(f"✅ Groq reflection generated ({len(reflection or "")} chars)")
        return reflection
        
    except Exception as e:
        logger.error(f"Groq reflection failed: {e}")
        return None

def generate_reflection_huggingface(query: str, verse: Dict) -> Optional[str]:
    """Generate reflection using Hugging Face Inference API (free)"""
    try:
        import httpx
        
        api_key = os.getenv("HF_API_KEY")
        if not api_key:
            logger.warning("HF_API_KEY not set, skipping Hugging Face")
            return None
        
        prompt = REFLECTION_PROMPT.format(
            query=query[:200],
            surah=verse["surah"],
            ayah=verse["ayah"],
            arabic=verse["arabic"][:100],
            translation=verse["translation_en"][:300],
            themes=", ".join(verse.get("tags", [])[:5])
        )
        
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.3,
                "return_full_text": False
            }
        }
        
        # Use a free model endpoint
        model_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(model_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                reflection = result[0]["generated_text"].strip()
                logger.info(f"✅ HF reflection generated ({len(reflection)} chars)")
                return reflection
        
        return None
        
    except Exception as e:
        logger.error(f"Hugging Face reflection failed: {e}")
        return None

def generate_reflection(query: str, verse: Dict, provider: str = "auto") -> Optional[tuple[str | None, str | None]]:
    """
    Generate reflection with automatic fallback between providers.
    
    Priority: Groq → Hugging Face → None
    """
    reflection = None
    reflection_provider = None
    cache_key = _get_cache_key(query, verse["surah"], verse["ayah"])
    if cache_key in _reflection_cache and _is_cache_valid(_reflection_cache[cache_key]):
        logger.info(f"📦 Serving cached reflection for {verse['surah']}:{verse['ayah']}")
        return _reflection_cache[cache_key]["reflection"]
    
    # Try providers in order
    providers = {
        "groq": generate_reflection_groq,
        "huggingface": generate_reflection_huggingface
    }
    
    if provider == "auto":
        # Try all in priority order
        for prov_name, prov_func in providers.items():
            reflection = prov_func(query, verse)
            reflection_provider = prov_name
            if reflection:
                break
    elif provider in providers:
        reflection = providers[provider](query, verse)
        reflection_provider = provider
    else:
        logger.warning(f"Unknown provider: {provider}")
        return None
    


    if reflection:
        _reflection_cache[cache_key] = {
            "reflection": reflection,
            "cached_at": datetime.now().isoformat(),
            "provider": provider
        }
        logger.info(f"💾 Cached reflection (TTL: 1 hour)")
    
    return (reflection, reflection_provider)

def get_cache_stats() -> Dict:
    """Return cache statistics for monitoring"""
    return {
        "cache_size": len(_reflection_cache),
        "cache_keys": list(_reflection_cache.keys())[:10]  # First 10 for debugging
    }
