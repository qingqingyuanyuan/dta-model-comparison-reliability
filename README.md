# Controlled Multi-Regime Evaluation for Drug–Target Affinity Prediction

Reproducibility materials accompanying the manuscript:

**Model Rankings in Drug–Target Affinity Prediction Vary Across Evaluation Regimes:
A Controlled Multi-Seed Study of Model-Comparison Reliability**

## Overview

This study evaluates the reliability of DTA model comparisons across three
generalization regimes:

- Warm
- Cold-target
- Cold-drug

Two benchmark datasets are used:

- KIBA
- Davis

Five methods are evaluated under a common protocol:

- ConcatFusion
- ProductFusion
- AttentionFusion
- DeepDTA-style
- GraphDTA-GCN-style

Three paired model seeds are used:

- 42
- 123
- 2026

The complete prespecified experimental matrix contains 126 training runs.

## Repository contents

- `src/` — source code
- `configs/` — experiment configurations
- `scripts/` — preprocessing and split-generation scripts
- `data/` — frozen split manifests and SHA256 hashes
- `results/` — frozen evaluation summaries
- `case_study/VEGFR2/` — exploratory post-hoc VEGFR2 analysis
- `audit/` — integrity-audit code and provenance materials
- `environment/` — software environment information

## Data

The original KIBA and Davis datasets are not redistributed here.
See `data/README_DATA.md`.

## Evaluation integrity

Formal data splits were frozen before training.

Model selection used validation data only. The held-out test sets were evaluated
in a separate final stage after completion of the prespecified training runs.

The study includes record-level and unique-input evaluation and an integrity
audit covering all 126 runs.

## Citation

Citation and permanent archive information will be added upon publication and
Zenodo archiving.

## Authors

- Haobo Kui — co-first author
- Xinyuan Cui — co-first author
- Yanfei Li — corresponding author

Haobo Kui and Xinyuan Cui contributed equally to this work.
