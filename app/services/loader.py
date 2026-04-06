import pandas as pd
import os
import logging

from scripts.clean_metadata import smart_split_labels
from scripts.generate_intent_map import clean_tags

logger = logging.getLogger(__name__)

# Module-level cache: survives across requests in the same worker process
_df_cache: pd.DataFrame | None = None

REQUIRED_COLUMNS = [
    "surah", "ayah", "arabic", "english", "urdu",
    "emotion", "tags", "category", "context"
]

def load_dataset(csv_path: str = "data/complete_quran_clean.csv") -> pd.DataFrame:
    global _df_cache
    if _df_cache is not None:
        return _df_cache

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'.\n"
            "Download from Kaggle and place it in the data/ folder."
            " and run the scripts/clean_metadata.py script to clean it before loading."
        )

    logger.info("Loading Quran dataset from disk...")
    df = pd.read_csv(csv_path)

    # Standardize column names (strip spaces, lowercase)
    df.columns = [c.strip().lower() for c in df.columns]

    # Validate structure
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Normalize multi-label string columns into clean Python lists
    for col in ["emotion", "tags", "context"]:
        # Fill NaN, convert to string, split by comma, strip whitespace, remove empty strings
        df[col] = df[col].fillna("").astype(str).apply(clean_tags)

    df['category'] = df['category'].fillna("").astype(str).apply(
        lambda x: [item.strip() for item in clean_tags(x) if item.strip()]
    )

    _df_cache = df
    logger.info(f"Dataset loaded & cached: {_df_cache.shape[0]} verses, {_df_cache.shape[1]} columns")
    return _df_cache
