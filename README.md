---
title: Bayyinah Quranic AI
emoji: 🕌
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# 🕌 Quran Ayah Recommender API

A guidance-aware API that matches emotional/textual inquiries to relevant Quranic verses.

## Features

- ✅ Hybrid semantic + metadata ranking
- ✅ Guidance-aware scoring (severity matching, diversity)
- ✅ Optional LLM-generated reflections
- ✅ Multi-language support (English, Urdu, Arabic)

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metrics` | GET | Service metrics |
| `/v2/recommend` | POST | Standard recommendation |
| `/v2/recommend-with-reflection` | POST | Recommendation + LLM reflection |

## Example Usage

```bash
curl -X POST https://YOUR_SPACE.hf.space/v2/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "I feel anxious about my future", "top_k": 3}'


