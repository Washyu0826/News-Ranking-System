# Experiment: Personalized Ranking vs Diversity-Aware Reranking

## Goal
Evaluate whether diversity-aware reranking improves feed diversity while maintaining ranking quality.

## Variants
A: Personalized LightGBM ranking  
B: Personalized LightGBM + Diversity reranking

## Metrics
NDCG@10, simulated CTR@10, Diversity@10, Coverage@10

## Expected Trade-off
Diversity may improve while CTR or NDCG may slightly decrease.
