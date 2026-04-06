from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=500, description="User's emotional or situational inquiry")
    top_k: int = Field(3, ge=1, le=10)
    language: str = Field("english", pattern="^(english|urdu|arabic)$")
    llm_provider: str = Field("auto", pattern="^(auto|groq|gemini|huggingface)$")

class VerseResponse(BaseModel):
    surah: int
    ayah: int
    text: str
    arabic: str
    translation_en: str
    translation_ur: str
    emotion: List[str]
    tags: List[str]
    category: List[str]
    context: List[str]
    relevance_score: float
    semantic_distance: float
    semantic_score: float
    metadata_boost: float
    severity_penalty: float
    repetition_penalty: float
    reflection: Optional[str] = None
    reflection_provider: Optional[str] = None

class RecommendResponse(BaseModel):
    query: str
    detected_emotions: List[str]
    verses: List[VerseResponse]
    metadata: dict
