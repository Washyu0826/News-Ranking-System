# Experiment: Popularity Baseline vs Personalized Ranking

## Goal
Evaluate whether personalized ranking improves news ordering quality.

## Hypothesis
H0: Personalized ranking does not improve ranking quality compared with popularity-based ranking.  
H1: Personalized ranking improves NDCG@10 and simulated CTR@10.

## Variants
A: Popularity + Freshness baseline  
B: Personalized LightGBM ranking

## Dataset
MIND impression logs

## Primary Metric
NDCG@10

## Secondary Metrics
MRR@10, Recall@10, simulated CTR@10, Diversity@10, latency

## Results
TBD

## Interpretation
TBD

## Risks
- Offline simulation may not fully represent real online user behavior
- Historical logs may contain position bias
- Click labels do not equal true user satisfaction

## Next Step
Add diversity-aware reranking and compare engagement-diversity trade-off.
