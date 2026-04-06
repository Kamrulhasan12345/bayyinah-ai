import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import json
import ast

def clean_tags(x):
    if isinstance(x, str):
        try:
            # Try to parse it as a real Python list first
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            # Fallback to splitting if it's just a normal comma-separated string
            return [t.strip(" []'\"") for t in x.split(",") if t.strip(" []'\"")]
    return x

def analyze_tag_cooccurrence(df, min_support=10):
    """
    Find which tags frequently appear together in verses.
    These form natural "guidance clusters".
    
    Args:
        df: Cleaned dataset
        min_support: Minimum times a tag pair must co-occur
    
    Returns:
        Dict of tag clusters with co-occurrence strength
    """
    # Count individual tag frequency
    all_tags = [tag for sublist in df["tags"] for tag in sublist]
    tag_freq = Counter(all_tags)
    
    # Count tag pair co-occurrence
    pair_counts = Counter()
    for tags in df["tags"]:
        if len(tags) >= 2:
            for pair in combinations(sorted(set(tags)), 2):
                pair_counts[pair] += 1
    
    # Filter to significant pairs
    significant_pairs = {
        pair: count for pair, count in pair_counts.items()
        if count >= min_support
    }
    
    # Build adjacency graph (which tags connect to which)
    tag_graph = defaultdict(set)
    for (tag1, tag2), count in significant_pairs.items():
        tag_graph[tag1].add(tag2)
        tag_graph[tag2].add(tag1)
    
    return {
        "tag_frequency": dict(tag_freq.most_common(50)),
        "significant_pairs": len(significant_pairs),
        "tag_graph_sample": {k: list(v)[:5] for k, v in list(tag_graph.items())[:10]}
    }

def auto_generate_intent_clusters(df, n_clusters=15):
    """
    Group tags into ~15 intent clusters based on co-occurrence patterns.
    Each cluster represents a spiritual need (e.g., "Trust in Hardship").
    """
    # Step 1: Get top tags by frequency
    all_tags = [tag for sublist in df["tags"] for tag in sublist]
    top_tags = [tag for tag, _ in Counter(all_tags).most_common(40)]
    
    # Step 2: For each top tag, find its most common co-occurring tags
    clusters = {}
    for seed_tag in top_tags[:n_clusters]:
        # Find verses containing this seed tag
        matching_verses = df[df["tags"].apply(lambda tags: seed_tag in tags)]
        
        # Count what other tags appear in these verses
        co_tags = Counter()
        for tags in matching_verses["tags"]:
            for tag in tags:
                if tag != seed_tag:
                    co_tags[tag] += 1
        
        # Build cluster: seed + top 3-5 co-occurring tags
        cluster_tags = [seed_tag] + [tag for tag, _ in co_tags.most_common(4)]
        clusters[seed_tag] = {
            "primary": seed_tag,
            "related": cluster_tags[1:],
            "verse_count": len(matching_verses),
            "all_tags": cluster_tags
        }
    
    return clusters

def generate_query_synonyms(clusters):
    """
    Auto-generate example user queries for each intent cluster
    using simple template expansion (no manual writing).
    """
    query_templates = {
        "Trust": ["I'm worried about", "uncertain about", "anxious about", "need trust for"],
        "Patience": ["I'm struggling with", "need patience for", "can't handle", "overwhelmed by"],
        "Forgiveness": ["I regret", "I sinned", "want forgiveness for", "feel guilty about"],
        "Guidance": ["I'm lost about", "need direction for", "confused about", "don't know how to"],
        "Hope": ["I feel hopeless about", "need hope for", "discouraged by", "lost faith in"],
        "Gratitude": ["want to be more grateful for", "thankful for", "blessed with"],
        "Justice": ["I was wronged", "need justice for", "treated unfairly", "saw injustice"],
        "Comfort": ["I'm grieving", "need comfort for", "feeling sad about", "lost someone"],
    }
    
    intent_queries = {}
    for cluster_name, cluster_data in clusters.items():
        # Find matching template (simple keyword match)
        matched_templates = []
        for template_key, templates in query_templates.items():
            if template_key.lower() in cluster_name.lower():
                matched_templates.extend(templates)
        
        # If no template match, generate generic queries
        if not matched_templates:
            matched_templates = [
                f"I need guidance about {cluster_name.lower()}",
                f"Help me with {cluster_name.lower()}",
                f"Quran about {cluster_name.lower()}"
            ]
        
        # Expand with cluster tags
        expanded = []
        for template in matched_templates[:3]:
            expanded.append(template)
            for tag in cluster_data["related"][:2]:
                expanded.append(f"{template} - {tag.lower()}")
        
        intent_queries[cluster_name] = expanded[:5]
    
    return intent_queries

if __name__ == "__main__":
    # Load cleaned dataset
    df = pd.read_csv("data/complete_quran_clean.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Ensure tags are parsed as lists (not strings)
    if isinstance(df["tags"].iloc[0], str):
        df["tags"] = df['tags'].apply(clean_tags)
    
    print("=" * 70)
    print("STEP 1: TAG CO-OCCURRENCE ANALYSIS")
    print("=" * 70)
    cooccurrence = analyze_tag_cooccurrence(df, min_support=5)
    print(f"Top tags: {list(cooccurrence['tag_frequency'].keys())[:10]}")
    print(f"Significant tag pairs: {cooccurrence['significant_pairs']}")
    
    print("\n" + "=" * 70)
    print("STEP 2: AUTO-GENERATE INTENT CLUSTERS")
    print("=" * 70)
    clusters = auto_generate_intent_clusters(df, n_clusters=15)
    
    for i, (seed, data) in enumerate(clusters.items(), 1):
        print(f"\n{i}. {seed}")
        print(f"   Related: {data['related']}")
        print(f"   Verses: {data['verse_count']}")
    
    print("\n" + "=" * 70)
    print("STEP 3: GENERATE EXAMPLE QUERIES")
    print("=" * 70)
    queries = generate_query_synonyms(clusters)
    
    # Save to JSON for use in API
    output = {
        "clusters": clusters,
        "example_queries": queries,
        "metadata": {
            "total_verses": len(df),
            "total_unique_tags": len(set(tag for sublist in df["tags"] for tag in sublist))
        }
    }
    
    with open("data/intent_map.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Intent map saved to data/intent_map.json")
    print(f"   - {len(clusters)} intent clusters")
    print(f"   - {sum(len(q) for q in queries.values())} example queries")
