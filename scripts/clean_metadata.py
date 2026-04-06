import pandas as pd
import re
import unicodedata
from collections import Counter

def normalize_unicode(text: str) -> str:
    if not text or text.strip() == "":
        return text;

    text = unicodedata.normalize('NFKC', text)

    quote_map = {
        '\'': "'",   # Left single curly quote
        '\'': "'",   # Right single curly quote
        '‘': "'",
        '’': "'",
        '"': '"',   # Left double curly quote
        '"': '"',   # Right double curly quote
        '`': "'",   # Backtick → apostrophe
    }

    for curly, straight in quote_map.items():
        text = text.replace(curly, straight)

    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    if text.endswith('"') or text.endswith("'"):
        text = text[:-1]
    
    # Step 4: Strip again after quote removal
    text = text.strip()
    
    return text

def smart_split_labels(label_string: str) -> list[str]:
    if not label_string or label_string.strip() == "":
        return []

    result = []
    current = ""
    paren_depth = 0

    for char in label_string:
        if char == '(':
            paren_depth += 1
            current += char
        elif char == ')':
            paren_depth -= 1
            current += char
        elif char == ',' and paren_depth == 0:
            if current.strip():
                result.append(current)
            current = ""
        else:
            current += char

    if current.strip():
        result.append(current.strip())

    return result

def clean_metadata(csv_path: str, output_path: str = None):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    simple_fields = ['emotion', 'tags', 'context']
    complex_fields = ['category']

    for field in complex_fields:
        if field not in df.columns:
            continue

        df[field] = df[field].fillna("").astype(str).apply(
                lambda x : [normalize_unicode(item.strip()) for item in smart_split_labels(x) if item.strip()]
        )

        typo_fixes = {
            "Supplication & Spiritivity (Dua, Dhikr, Tazkiyah)": "Supplication & Spirituality (Dua, Dhikr, Tazkiyah)",
            "Supplication & Spiritality (Dua, Dhikr, Tazkiyah)": "Supplication & Spirituality (Dua, Dhikr, Tazkiyah)",
            "Ethics & Morality (Akhlaak)": "Ethics & Morality (Akhlaq)",
            "Eschatology": "Eschatology (Akhirah)",
            "Divine Attributes & Signs": "Divine Attributes & Signs (Asma wa Sifat)"
        }

  # Eschatology (Akhirah): 1698
  # History & Stories (Qasas al-Anbiya): 1289
  # Faith (Aqeedah): 1142
  # Divine Attributes & Signs (Asma wa Sifat): 1033
  # Ethics & Morality (Akhlaq): 505
  # Supplication & Spirituality (Dua, Dhikr, Tazkiyah): 292
  # Worship ('Ibadah): 162
  # Law (Ahkam): 143
  # Social Relations (Mu'amalat): 57
  # Supplication & Spiritality (Dua, Dhikr, Tazkiyah): 4
  # Ethics & Morality (Akhlaak): 3
  # Moral teaching context: 2
  # Worship ('Ibadah)': 2
  # Spiritual reminder: 2
  # Revelation: 2
  # Divine Attributes & Signs: 1
  # Eschatology: 1
  # Supplication & Spiritivity (Dua, Dhikr, Tazkiyah): 1


        df[field] = df[field].apply(
                lambda items: [typo_fixes.get(item, item) for item in items]
        )

        all_items = [item for sublist in df[field] for item in sublist]
        freq = Counter(all_items)
        print(f"\n{field.upper()} - Most Common values:")
        for item, count in freq.most_common():
            print(f"  {item}: {count}")

    for field in simple_fields:
        if field not in df.columns:
            continue

        df[field] = df[field].fillna("").astype(str).apply(
                lambda x : [normalize_unicode(item.strip()) for item in x.split(",") if item.strip()]
        )

        typo_fixes = {
                "Eschological Context": "Eschatological Context",
                "Divine Decree": "Divine Decree",
                "Qur'an": "Quran",
        }

        df[field] = df[field].apply(
                lambda items: [typo_fixes.get(item, item) for item in items]
        )

        all_items = [item for sublist in df[field] for item in sublist]
        freq = Counter(all_items)
        print(f"\n{field.upper()} - Top 10 values:")
        for item, count in freq.most_common():
            print(f"  {item}: {count}")

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\n✅ Cleaned dataset saved to {output_path}")

    return df

if __name__  == "__main__":
    df_clean = clean_metadata("data/Complete_Quran_data.csv", "data/complete_quran_clean.csv")
