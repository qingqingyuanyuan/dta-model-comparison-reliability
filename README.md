# Controlled Multi-Regime Evaluation for Drug–Target Affinity Prediction

Reproducibility materials accompanying **Model Rankings in Drug–Target Affinity Prediction Vary Across Evaluation Regimes: A Controlled Multi-Seed Study of Model-Comparison Reliability**.

## Current revision — v1.1.0

The current manuscript evidence comprises **282 runs**, kept in distinct layers:

| Evidence layer | Runs | Scope |
|---|---:|---|
| Original campaign | 126 | KIBA/Davis primary comparisons and prespecified sensitivity/efficiency runs |
| Capacity controls | 36 | Original splits; widened DeepDTA-style/GraphDTA-GCN-style baselines; matched AdamW protocol |
| Second KIBA split | 30 | Five configurations × two cold-start regimes × three model seeds |
| GPCR extension | 30 | One human GPCR Ki cohort/split; within-cohort cold-drug/cold-target retraining |
| Protease extension | 60 | Two splits of one human protease Ki cohort; within-cohort retraining |

Historical 36-run Adam exploration is auxiliary and **excluded** from the 282-run total. Paired model seeds are 42, 123 and 2026. Warm evaluation belongs to KIBA/Davis; external extensions cover cold-drug and full-sequence-defined cold-target evaluation.

The original KIBA mean-ordering reversal was conditional on training, model set and split. Cross-regime mean-ranking Kendall tau is -0.60 for GPCR and +0.40/+0.20 for the two protease splits. Two non-kinase classes do not establish proteome-wide generality. GPCR cold-target contains only seven independent test sequences. Protease splits share two test sequences and 163 cold-drug entities; they are not two independent external cohorts. External capacities are unequal, so these runs cannot isolate causal superiority of fusion mechanisms. This is retraining within cohorts, not zero-shot transfer of kinase-trained weights.

## Find the new files

- [Revision guide](revisions/v1.1.0/README.md): scope, reconstruction, checks and known gaps.
- [External metrics](revisions/v1.1.0/results): 90-run recomputed metrics, summaries, partitions and rank correlations.
- [Final table exports](revisions/v1.1.0/final_table_exports): corrected document tables; source labels/structures are projected out.
- [Preparation and evaluation code](revisions/v1.1.0/code/external_90): readable external campaign entry points; complete modules/configurations are in the payload.
- [Payload index](revisions/v1.1.0/payload_index.json): checksummed ZIP parts containing the original-126 prediction projections, 36 AdamW controls, 30 second-split results, external-90 results, code, metadata and frozen integer membership.
- Existing `src`, `configs`, `data`, `results`, `case_study`, `audit` and `environment` directories retain the earlier published snapshot. Their historical 126-run scope must not be mistaken for the expanded revision.

```bash
python scripts/unpack_revision_v110.py --output ../dta_v110_materials
```

All ZIP parts unpack into one output directory. The script checks hashes, rejects unsafe paths and does not load weights, execute archived code or run training. No third-party source structures, sequences or affinity labels are redistributed. Reconstruct labels/features from independently acquired upstream versions and exact membership/hashes, as explained in the payload provenance documentation.

Weights and large execution logs are omitted from the lightweight public payload. Complete original-126 weights/logs/record predictions and original bootstrap RNG states were not retained. Historical checkpoint/prediction receipts refer to the original files, not the derived public projections. Inherited bootstrap summaries are distinguished from independently recomputed metrics; file verification does not reconstruct historical test access or prove training determinism.

## Final statistical corrections

- S17 errors use raw affinity units (MSE uses squared raw units).
- S13 Delta = C-index(A) - C-index(B), positive in favor of A.
- S18 contains the mean and sample SD of three paired differences; its bootstrap contrasts are elsewhere.
- S22 now uses sample SD (ddof=1); original per-seed scores and means are unchanged.

## Authors and citation

**Haobo Kui; Xinyuan Cui**, in that order. Corresponding author: **Haobo Kui**, School of Chemistry and Chemical Engineering, Harbin Normal University; 2993272783@qq.com. Xinyuan Cui: School of Physics and Electronic Engineering, Harbin Normal University. No equal-contribution designation is asserted in the current metadata.

Previous archives remain available: [v1.0.0](https://doi.org/10.5281/zenodo.22860429), [v1.0.1](https://doi.org/10.5281/zenodo.24051010). The [concept DOI](https://doi.org/10.5281/zenodo.22860428) covers the existing version series. A version-specific DOI for this expanded revision will be cited only after the record is verified. Publishing GitHub files alone does not establish that Zenodo has ingested them. See [archive status](docs/ARCHIVE_STATUS.md).

## License scope

Previously published v1.0.0/v1.0.1 material retains its original CC BY 4.0 grant. New revision additions retain the supplied public candidate's CC BY-NC 4.0 terms; third-party data and dependencies keep their own terms. See [LICENSE_SCOPE.md](LICENSE_SCOPE.md). Peer-review correspondence, cover letters and private submission documents are not public release assets.
