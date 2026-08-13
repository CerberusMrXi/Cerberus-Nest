# Experiments

## E001 — Baseline

Purpose: establish reproducible baseline behavior.

Parameters:
- seed: 42
- steps: 5000
- curiosity weight: 0.5
- social interaction: enabled

Output:
`logs/baseline_42.csv`

## E002 — Curiosity Ablation

Set curiosity weight to zero.

Compare:
- exploration
- reward
- memory growth
- survival

## E003 — Social Ablation

Disable caregiver interaction.

Compare against baseline.

## E004 — Memory Ablation

Reduce memory capacity substantially.

Measure long-term behavioral changes.

## E005 — World Model Ablation

Disable world-model learning.

Measure prediction and planning-related behavior.

All experiments should use fixed seeds for reproducibility.
