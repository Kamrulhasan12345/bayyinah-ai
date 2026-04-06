from fastapi import FastAPI, HTTPException, Query
import logging
from collections import Counter
from dotenv import load_dotenv

from app.models import RecommendRequest, RecommendResponse
from app.services.embeddings import get_model
from app.services.llm_reflection import get_cache_stats
from app.services.loader import load_dataset
from app.services.recommender import recommend_verses
from app.services.llm_reflection import generate_reflection

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Bayyinah Quranic AI", version="2.0")

@app.on_event("startup")
async def startup_event():
    global _model, _faiss_index, _df_cache

    try:
        logger.info("Starting up Bayyinah Quranic AI...")
        _model = get_model()
        _df_cache = load_dataset()
        logger.info("Startup complete. Model and dataset are ready.")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)


@app.get("/health")
async def health():
    return {'status': 'ok', 'message': 'Bayyinah Quranic AI is healthy and ready to serve!'}

@app.post("/v2/recommend-with-reflection")
async def recommend_with_reflection(req: RecommendRequest, llm_provider: str = Query("auto", description="LLM provider for reflection: auto, groq, huggingface")):
    """
    v2 Recommendation + LLM-generated reflection.
    
    Returns verses with compassionate, contextual reflections for each.
    Uses free LLM APIs with automatic fallback.
    """
    from app.services.recommender import recommend_verses
    
    try:
        # Step 1: Get standard recommendations
        rec_result = recommend_verses(query=req.query, top_k=req.top_k, language=req.language)
        
        # Step 2: Generate reflection for each verse
        verses_with_reflection = []
        for verse in rec_result["verses"]:
            verse_dict = verse.dict() if hasattr(verse, "dict") else verse
            
            reflection = generate_reflection(
                query=req.query,
                verse=verse_dict,
                provider=llm_provider
            )

            if reflection:
                verse_dict["reflection"] = reflection[0]
                verse_dict["reflection_provider"] = reflection[1]
            else:
                verse_dict["reflection"] = None
                verse_dict["reflection_provider"] = None
            
            verses_with_reflection.append(verse_dict)
        
        return {
            "query": req.query,
            "detected_emotions": rec_result["detected_emotions"],
            "verses": verses_with_reflection,
            "metadata": {
                **rec_result["metadata"],
                "reflection_enabled": True,
                "llm_provider": llm_provider,
                "reflections_generated": sum(1 for v in verses_with_reflection if v["reflection"])
            }
        }
        
    except Exception as e:
        logger.error(f"Reflection endpoint failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate reflections. Try again or disable reflection."
        )

# Add cache stats endpoint for monitoring
@app.get("/debug/reflection-cache")
async def reflection_cache_stats():
    """View LLM reflection cache status"""
    return get_cache_stats()

@app.post("/v2/recommend", response_model=RecommendResponse)
async def get_recommendation(req: RecommendRequest):
    try:
        result = recommend_verses(query=req.query, top_k=req.top_k, language=req.language)
        return RecommendResponse(**result)
    except Exception as e:
        logger.error(f"Recommendation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during recommendation.")

@app.get("/data/info")
async def dataset_info():
    """Test endpoint to verify dataset loading & structure"""
    try:
        df = load_dataset()
        return {
            "total_verses": len(df),
            "columns": list(df.columns),
            "sample_emotions": df["emotion"].head(5).tolist(),
            "memory_cached": True
        }
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/metadata-stats")
async def metadata_stats():
    df = load_dataset()
    stats = {}

    for field in ["tags", "emotion", "context", "category"]:
        if field in df.columns:
            all_items = [item for sublist in df[field] for item in sublist]
            freq = Counter(all_items)
            stats[field] = {
                'unique_count': len(freq),
                'top_5': freq.most_common(5)
            }

    return stats
