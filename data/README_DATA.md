# Data and frozen split availability

This study uses the KIBA and Davis drug–target affinity benchmark datasets.

The original KIBA and Davis datasets are third-party research resources and are
not redistributed in this repository. Users should obtain the source datasets
from their original publications or official distribution sources.

## Public reproducibility materials

This directory provides:

- `frozen_split_indices.csv` — source-row membership of the frozen formal splits;
- `frozen_split_manifest.csv` — row counts and partition metadata;
- `FROZEN_SPLIT_INDEX_SHA256.txt` — hashes of the public split-index artifacts;
- `FORMAL_SPLIT_SOURCE_SHA256.txt` — hashes of the 18 frozen formal split CSVs
  used in the study.

The public split index contains no molecular SMILES strings, protein sequences,
or affinity labels.

The frozen split-generation implementation is provided in:

`../src/FINAL_TEST_WORKSPACE/scripts/prepare_final_datasets.py`

## Formal evaluation regimes

### Warm

Seen-entity interpolation. Identical model inputs were kept within a single
partition.

### Cold-drug

Drug identity was defined using canonical SMILES, with zero canonical-SMILES
overlap across train, validation, and test partitions.

### Cold-target

Target identity was defined using the full protein sequence, with zero
protein-sequence overlap across train, validation, and test partitions.

The split seed was fixed at 42.

## Frozen partition sizes

| Dataset | Setting | Train | Validation | Test |
|---|---|---:|---:|---:|
| KIBA | Warm | 94,429 | 11,804 | 11,803 |
| KIBA | Cold-target | 85,975 | 12,421 | 19,640 |
| KIBA | Cold-drug | 94,491 | 11,745 | 11,800 |
| Davis | Warm | 24,051 | 3,000 | 3,005 |
| Davis | Cold-target | 22,372 | 2,856 | 4,828 |
| Davis | Cold-drug | 24,310 | 3,094 | 2,652 |
