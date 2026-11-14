# v1.1.0 — Expanded revision evidence and corrected statistical reporting

Adds materials for the 282-run evidence set: original 126, 36 matched-optimizer capacity controls, 30 second-split KIBA runs and 90 within-cohort GPCR/protease runs. The separate 36-run Adam exploration remains excluded.

Includes public prediction projections, source code/configurations, histories, provenance, partition membership/hashes, numerical/hash checks and corrected table exports. The independently checksummed payload ZIP parts are stored in the tagged tree itself. Use `python scripts/unpack_revision_v110.py --output ../dta_v110_materials` to reconstruct the directory layout.

Corrects S17 raw-error units, S13 contrast direction, S18 mean/SD description and S22 sample SD. Updates author order/affiliations and corresponding-author information without asserting equal contribution. Original scientific per-seed scores are preserved.

Limits: only two non-kinase target classes; seven GPCR cold-target proteins; two protease splits of one cohort sharing two test sequences and 163 cold-drug entities; unequal external capacities. No universal winner/reversal or zero-shot family transfer is established. Third-party source data, weights and large logs are omitted from the lightweight archive; complete original-126 artifacts and original bootstrap RNG states were not retained. Historical receipts and public projection hashes have different scopes.

New additions follow the supplied CC BY-NC 4.0 candidate terms; previous v1.0.0/v1.0.1 CC BY 4.0 grants remain unchanged. Zenodo publication is reported separately after verification; no unassigned DOI is cited.
