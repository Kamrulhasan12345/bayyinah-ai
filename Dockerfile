# Dockerfile (HF Spaces optimized)
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory (HF Spaces persists /tmp better than root)
RUN mkdir -p /tmp/quran-data
ENV EMBEDDINGS_CACHE_PATH=/tmp/quran-data/verse_embeddings.npy
ENV INTENT_MAP_PATH=/tmp/quran-data/intent_map.json

# Pre-download embedding model (avoids cold start)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Expose port (HF Spaces uses 7860 by default)
EXPOSE 7860

# Run with production settings
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
