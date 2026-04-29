# NewsRank: A Personalized News Ranking System

## 1) Project Overview
NewsRank is a **production-style ML ranking system** for personalized news feed ranking, not just a model training exercise.

## 2) Why News Ranking?
News feeds require balancing relevance, freshness, popularity, and diversity under ranking constraints.

## 3) System Architecture
Dataset Loader → Feature Engineering → Candidate Generation → Ranking → Diversity Reranking → Evaluation/A-B Simulation → API/Dashboard.

## 4) Dataset
Primary target dataset: **MIND-small** (`news.tsv`, `behaviors.tsv`). If missing, the project runs using a built-in toy dataset generator.

## 5) Feature Engineering
- Article: category/subcategory, title length, abstract length, has_abstract, freshness, popularity.
- User: click count, history length, top categories, category preference.
- Interaction: category match, title overlap, similarity fallback, popularity/freshness/activity scores.

## 6) Candidate Generation
- Popularity-based Top-N retrieval.
- Embedding retrieval via TF-IDF + cosine similarity (FAISS/sentence-transformers extensible).

## 7) Ranking Models
- Baseline weighted ranker (MVP).
- LightGBM ranker interface (fallback-safe).

## 8) Diversity Reranking
Applies freshness bonus and penalties for duplicate category/source to improve feed diversity without fully overriding relevance.

## 9) Offline A/B Testing Simulation
Historical impressions are replayed to compare baseline vs personalized and personalized vs diversity-reranked variants, with bootstrap confidence intervals.

## 10) Evaluation Metrics
NDCG@K, MRR@K, Recall@K, Precision@K, simulated CTR@K, Diversity@K, Coverage@K, AUC, and latency tracking hooks.

## 11) API Usage
Run:
```bash
uvicorn src.api.main:app --reload
```
Endpoints:
- `GET /health`
- `POST /rank`
- `POST /recommend`
- `GET /metrics`

## 12) Dashboard
Run:
```bash
streamlit run src/dashboard/app.py
```

## 13) Experiment Results
Templates live in `experiments/` and can be populated after runs.

## 14) Error Analysis
Use grouped metric breakdown by user segments/categories and inspect reranking penalties.

## 15) Future Work
- Train/tune full LightGBM lambdarank on MIND-small.
- Integrate FAISS + sentence-transformers.
- Add feature store and online inference cache.

---
### MVP Status
Implemented MVP includes:
1. Toy dataset generator
2. Baseline ranker
3. Evaluation metrics
4. Offline A/B helper utilities
5. Diversity reranker
6. FastAPI `/health` and `/rank`
7. This README draft
