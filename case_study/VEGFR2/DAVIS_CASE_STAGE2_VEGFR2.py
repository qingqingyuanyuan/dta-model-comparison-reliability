import csv
import json
import glob
import os
import re
import math
import hashlib
import pickle
from collections import defaultdict, Counter

# ============================================================
# CONFIG
# ============================================================

TARGET = "VEGFR2"

BASE = "./FINAL_TEST_WORKSPACE/evaluation_outputs"
RAW_DIR = "./FINAL_TEST_WORKSPACE/data/raw/davis"

TARGETS_FILE = os.path.join(RAW_DIR, "targets.txt")
DRUGS_FILE = os.path.join(RAW_DIR, "drugs.txt")
Y_FILE = os.path.join(RAW_DIR, "Y.txt")

LOCK_FILE = "./DAVIS_CASE_TARGET_LOCK.txt"

OUTDIR = "./DAVIS_CASE_VEGFR2"
os.makedirs(OUTDIR, exist_ok=True)

EXPECTED_SEEDS = {42, 123, 2026}

MODEL_ORDER = [
    "ConcatFusion",
    "ProductFusion",
    "AttentionFusion",
    "DeepDTA-style",
    "GraphDTA-GCN-style",
]

# ============================================================
# HELPERS
# ============================================================

def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def classify_file(path):
    b = os.path.basename(path)

    if "deepdta_style__" in b:
        model = "DeepDTA-style"
    elif "graphdta_gcn_style__" in b:
        model = "GraphDTA-GCN-style"
    elif "davis_cold_target_concat_seed" in b:
        model = "ConcatFusion"
    elif "davis_cold_target_product_seed" in b:
        model = "ProductFusion"
    elif "davis_cold_target_attention_seed" in b:
        model = "AttentionFusion"
    else:
        raise RuntimeError("Cannot classify model file: " + path)

    m = re.search(r"seed(42|123|2026)", b)
    if not m:
        raise RuntimeError("Cannot identify seed: " + path)

    return model, int(m.group(1))

def frozen_key(r):
    return (
        r["Source Split"],
        r["Source Row"],
        r["Target Sequence"],
        r["Canonical SMILES"],
        r["Label"],
    )

def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

def kendall_tau_b(x, y):
    """
    Kendall tau-b without scipy.
    Handles ties in x and y.
    """
    if len(x) != len(y):
        raise ValueError("x/y length mismatch")

    C = 0
    D = 0
    Tx = 0
    Ty = 0
    Tboth = 0

    n = len(x)

    for i in range(n - 1):
        for j in range(i + 1, n):
            sx = sign(x[i] - x[j])
            sy = sign(y[i] - y[j])

            if sx == 0 and sy == 0:
                Tboth += 1
            elif sx == 0:
                Tx += 1
            elif sy == 0:
                Ty += 1
            elif sx == sy:
                C += 1
            else:
                D += 1

    denom = math.sqrt((C + D + Tx) * (C + D + Ty))

    tau = (C - D) / denom if denom else float("nan")

    return tau, {
        "concordant": C,
        "discordant": D,
        "ties_observed_only": Tx,
        "ties_prediction_only": Ty,
        "ties_both": Tboth,
    }

def rmse(obs, pred):
    return math.sqrt(
        sum((a - b) ** 2 for a, b in zip(obs, pred)) / len(obs)
    )

def min_rank(values, higher_is_better=True):
    """
    Competition/min rank:
    1,2,2,4...
    """
    counts = Counter(values.values())
    unique = sorted(
        counts.keys(),
        reverse=higher_is_better
    )

    value_to_rank = {}
    rank = 1

    for v in unique:
        value_to_rank[v] = rank
        rank += counts[v]

    return {k: value_to_rank[v] for k, v in values.items()}

def top_k_keys(values, secondary, k=10, higher_is_better=True):
    if higher_is_better:
        ordered = sorted(
            values,
            key=lambda key: (-values[key], secondary[key])
        )
    else:
        ordered = sorted(
            values,
            key=lambda key: (values[key], secondary[key])
        )

    return ordered[:k]

def boundary_tie_info(values, k=10, higher_is_better=True):
    vals = sorted(values.values(), reverse=higher_is_better)

    cutoff = vals[k - 1]

    if higher_is_better:
        strictly_better = sum(v > cutoff for v in vals)
    else:
        strictly_better = sum(v < cutoff for v in vals)

    tied_at_cutoff = sum(v == cutoff for v in vals)

    ambiguous = strictly_better < k < strictly_better + tied_at_cutoff

    return cutoff, tied_at_cutoff, ambiguous

# ============================================================
# 0. REQUIRE TARGET LOCK
# ============================================================

if not os.path.exists(LOCK_FILE):
    raise RuntimeError(
        "Target lock file is missing. Create DAVIS_CASE_TARGET_LOCK.txt first."
    )

with open(LOCK_FILE, "r", encoding="utf-8") as f:
    lock_text = f.read()

if TARGET not in lock_text:
    raise RuntimeError(
        f"Lock file does not contain selected target {TARGET}."
    )

print("=" * 90)
print("TARGET LOCK")
print("=" * 90)
print("Selected target:", TARGET)
print("Lock SHA256:", sha256_file(LOCK_FILE))

# ============================================================
# 1. FIND 15 FORMAL FILES
# ============================================================

files = sorted(
    p for p in glob.glob(
        BASE + "/**/*davis*cold_target*test_predictions.csv",
        recursive=True
    )
    if "/sensitivity/" not in p
)

if len(files) != 15:
    raise RuntimeError(
        f"Expected 15 formal files, found {len(files)}"
    )

file_map = {}

for p in files:
    model, seed = classify_file(p)

    key = (model, seed)
    if key in file_map:
        raise RuntimeError("Duplicate model/seed file: " + str(key))

    file_map[key] = p

for model in MODEL_ORDER:
    seeds = {
        seed for (m, seed) in file_map
        if m == model
    }

    if seeds != EXPECTED_SEEDS:
        raise RuntimeError(
            f"{model} seeds are {seeds}, expected {EXPECTED_SEEDS}"
        )

print()
print("=" * 90)
print("FORMAL INPUT FILE CHECK")
print("=" * 90)
print("Formal prediction files:", len(files))
print("Models:", len(MODEL_ORDER))
print("Seeds/model:", sorted(EXPECTED_SEEDS))

# ============================================================
# 2. READ TARGET-SPECIFIC FROZEN ROWS
# ============================================================

metadata = {}
predictions = defaultdict(lambda: defaultdict(dict))
reference_keys = None
target_sequences = set()

for (model, seed), p in sorted(file_map.items()):
    rows = read_csv(p)

    target_rows = [
        r for r in rows
        if r["Target ID"] == TARGET
    ]

    if not target_rows:
        raise RuntimeError(
            f"{TARGET} not found in {p}"
        )

    keys = set()

    for r in target_rows:
        k = frozen_key(r)
        keys.add(k)

        target_sequences.add(r["Target Sequence"])

        current_meta = {
            "Target ID": r["Target ID"],
            "Target Sequence": r["Target Sequence"],
            "Canonical SMILES": r["Canonical SMILES"],
            "SMILES raw": r["SMILES raw"],
            "Label": float(r["Label"]),
            "Source Split": r["Source Split"],
            "Source Row": r["Source Row"],
        }

        if k in metadata:
            old = metadata[k]
            for fld in current_meta:
                if str(old[fld]) != str(current_meta[fld]):
                    raise RuntimeError(
                        f"Metadata mismatch for {k}: {fld}"
                    )
        else:
            metadata[k] = current_meta

        predictions[model][k][seed] = float(r["Prediction_raw"])

    if reference_keys is None:
        reference_keys = keys
    elif keys != reference_keys:
        raise RuntimeError(
            f"Target-specific record mismatch in {p}"
        )

    print(
        f"{model:<22} seed={seed:<4} "
        f"target rows={len(target_rows)}"
    )

if len(target_sequences) != 1:
    raise RuntimeError(
        f"{TARGET} maps to {len(target_sequences)} sequences "
        "in final frozen predictions."
    )

keys = sorted(
    reference_keys,
    key=lambda k: (
        metadata[k]["Source Split"],
        int(metadata[k]["Source Row"])
    )
)

print()
print("Final target rows:", len(keys))
print(
    "Unique canonical compounds:",
    len({metadata[k]["Canonical SMILES"] for k in keys})
)
print("Unique target sequences:", len(target_sequences))

if len(keys) != 68:
    print(
        "WARNING: Expected 68 Davis compounds but found",
        len(keys)
    )

# ============================================================
# 3. VERIFY TARGET METADATA MAPPING
# ============================================================

with open(TARGETS_FILE, "r", encoding="utf-8") as f:
    target_map = json.load(f)

selected_seq = next(iter(target_sequences))

if TARGET not in target_map:
    raise RuntimeError(
        f"{TARGET} absent from targets.txt"
    )

if target_map[TARGET] != selected_seq:
    raise RuntimeError(
        "VEGFR2 target sequence does not match targets.txt"
    )

same_seq_names = sorted(
    name for name, seq in target_map.items()
    if seq == selected_seq
)

print()
print("=" * 90)
print("TARGET METADATA CHECK")
print("=" * 90)
print("Target:", TARGET)
print("Sequence length:", len(selected_seq))
print("Names sharing same sequence:", same_seq_names)

if same_seq_names != [TARGET]:
    raise RuntimeError(
        "Selected target does not have a clean single-name sequence mapping."
    )

# ============================================================
# 4. RAW DAVIS LABEL-DIRECTION AUDIT
# ============================================================

# Davis raw data are used only to verify the meaning/direction
# of the already-frozen Label column. No model selection occurs.

with open(DRUGS_FILE, "r", encoding="utf-8") as f:
    drug_map = json.load(f)

with open(Y_FILE, "rb") as f:
    Y = pickle.load(f, encoding="latin1")

drug_names = list(drug_map.keys())
target_names = list(target_map.keys())

if len(Y) != len(drug_names):
    raise RuntimeError(
        f"Unexpected Y rows: {len(Y)} vs drugs {len(drug_names)}"
    )

if any(len(row) != len(target_names) for row in Y):
    raise RuntimeError(
        "Unexpected Y matrix dimensions."
    )

target_index = target_names.index(TARGET)

raw_smiles_to_indices = defaultdict(list)
for i, name in enumerate(drug_names):
    raw_smiles_to_indices[drug_map[name]].append(i)

label_audit_diffs = []
drug_name_by_key = {}

for k in keys:
    raw_smi = metadata[k]["SMILES raw"]

    idxs = raw_smiles_to_indices.get(raw_smi, [])

    if len(idxs) == 1:
        i = idxs[0]
        kd_nm = float(Y[i][target_index])

        if kd_nm <= 0:
            raise RuntimeError("Non-positive Davis Kd encountered.")

        # Davis pKd-like transformation:
        # pKd = -log10(Kd [M])
        #     = 9 - log10(Kd [nM])
        transformed = 9.0 - math.log10(kd_nm)

        diff = abs(
            transformed - metadata[k]["Label"]
        )
        label_audit_diffs.append(diff)

        drug_name_by_key[k] = drug_names[i]

    else:
        drug_name_by_key[k] = ""

if not label_audit_diffs:
    raise RuntimeError(
        "Could not audit Davis label direction against raw Y.txt."
    )

max_label_diff = max(label_audit_diffs)

print()
print("=" * 90)
print("RAW LABEL-DIRECTION AUDIT")
print("=" * 90)
print(
    "Matched compounds:",
    len(label_audit_diffs),
    "/",
    len(keys)
)
print(
    "Max |Label - (9-log10(Kd_nM))|:",
    f"{max_label_diff:.12g}"
)

if max_label_diff > 1e-6:
    raise RuntimeError(
        "Frozen Label does not match expected Davis pKd-like transformation. "
        "Stop before defining Top-10 direction."
    )

HIGHER_IS_BETTER = True

print(
    "Direction confirmed: larger frozen Label = stronger affinity."
)

# ============================================================
# 5. THREE-SEED MEAN PREDICTIONS
# ============================================================

mean_predictions = {}

for model in MODEL_ORDER:
    mean_predictions[model] = {}

    for k in keys:
        seed_map = predictions[model][k]

        if set(seed_map.keys()) != EXPECTED_SEEDS:
            raise RuntimeError(
                f"Missing seeds for {model}, record {k}"
            )

        mean_predictions[model][k] = (
            sum(seed_map.values()) /
            len(EXPECTED_SEEDS)
        )

observed = {
    k: metadata[k]["Label"]
    for k in keys
}

secondary = {
    k: metadata[k]["Canonical SMILES"]
    for k in keys
}

# ============================================================
# 6. MAIN CASE-STUDY METRICS
# ============================================================

observed_top10 = top_k_keys(
    observed,
    secondary,
    k=10,
    higher_is_better=HIGHER_IS_BETTER
)

obs_cutoff, obs_tied, obs_ambiguous = boundary_tie_info(
    observed,
    k=10,
    higher_is_better=HIGHER_IS_BETTER
)

summary_rows = []

print()
print("=" * 90)
print("VEGFR2 CASE-STUDY RESULTS: 3-SEED MEAN")
print("=" * 90)

for model in MODEL_ORDER:
    pred = mean_predictions[model]

    obs_vec = [observed[k] for k in keys]
    pred_vec = [pred[k] for k in keys]

    tau, tau_details = kendall_tau_b(
        obs_vec,
        pred_vec
    )

    predicted_top10 = top_k_keys(
        pred,
        secondary,
        k=10,
        higher_is_better=HIGHER_IS_BETTER
    )

    overlap = len(
        set(observed_top10) &
        set(predicted_top10)
    )

    pred_cutoff, pred_tied, pred_ambiguous = boundary_tie_info(
        pred,
        k=10,
        higher_is_better=HIGHER_IS_BETTER
    )

    r = rmse(obs_vec, pred_vec)

    row = {
        "Target": TARGET,
        "N_compounds": len(keys),
        "Model": model,
        "Kendall_tau_b": tau,
        "Top10_overlap_count": overlap,
        "Top10_overlap_fraction": overlap / 10.0,
        "RMSE_raw": r,
        "Observed_top10_cutoff": obs_cutoff,
        "Observed_cutoff_tied_count": obs_tied,
        "Observed_top10_boundary_ambiguous": obs_ambiguous,
        "Predicted_top10_cutoff": pred_cutoff,
        "Predicted_cutoff_tied_count": pred_tied,
        "Predicted_top10_boundary_ambiguous": pred_ambiguous,
        **tau_details,
    }

    summary_rows.append(row)

    print(
        f"{model:<22} "
        f"tau-b={tau: .4f}   "
        f"Top10={overlap}/10   "
        f"RMSE={r:.4f}"
    )

print()
print(
    "Observed Top-10 cutoff:",
    obs_cutoff,
    "| tied at cutoff:",
    obs_tied,
    "| ambiguous:",
    obs_ambiguous
)

# ============================================================
# 7. SEED-WISE SUPPLEMENTARY RESULTS
# ============================================================

seedwise_rows = []

for model in MODEL_ORDER:
    for seed in sorted(EXPECTED_SEEDS):

        pred = {
            k: predictions[model][k][seed]
            for k in keys
        }

        obs_vec = [observed[k] for k in keys]
        pred_vec = [pred[k] for k in keys]

        tau, _ = kendall_tau_b(
            obs_vec,
            pred_vec
        )

        predicted_top10 = top_k_keys(
            pred,
            secondary,
            k=10,
            higher_is_better=HIGHER_IS_BETTER
        )

        overlap = len(
            set(observed_top10) &
            set(predicted_top10)
        )

        seedwise_rows.append({
            "Target": TARGET,
            "Model": model,
            "Seed": seed,
            "Kendall_tau_b": tau,
            "Top10_overlap_count": overlap,
            "Top10_overlap_fraction": overlap / 10.0,
            "RMSE_raw": rmse(
                obs_vec,
                pred_vec
            ),
        })

# ============================================================
# 8. RANK TABLES
# ============================================================

observed_rank = min_rank(
    observed,
    higher_is_better=True
)

mean_ranks = {
    model: min_rank(
        mean_predictions[model],
        higher_is_better=True
    )
    for model in MODEL_ORDER
}

all_rows = []

for k in keys:
    row = {
        "Drug": drug_name_by_key.get(k, ""),
        "Target": TARGET,
        "SourceSplit": metadata[k]["Source Split"],
        "SourceRow": metadata[k]["Source Row"],
        "Canonical_SMILES": metadata[k]["Canonical SMILES"],
        "SMILES_raw": metadata[k]["SMILES raw"],
        "Observed_Label": observed[k],
        "Observed_Rank": observed_rank[k],
    }

    for model in MODEL_ORDER:
        row[f"{model}_MeanPrediction"] = (
            mean_predictions[model][k]
        )
        row[f"{model}_Rank"] = (
            mean_ranks[model][k]
        )

        for seed in sorted(EXPECTED_SEEDS):
            row[f"{model}_Seed{seed}"] = (
                predictions[model][k][seed]
            )

    all_rows.append(row)

all_rows.sort(
    key=lambda r: (
        r["Observed_Rank"],
        r["Canonical_SMILES"]
    )
)

top10_rows = [
    r for r in all_rows
    if r["Observed_Rank"] <= 10
]

# Exact 10 rows in deterministic observed order
top10_exact_keys = set(observed_top10)
top10_exact_rows = [
    r for r in all_rows
    if any(
        metadata[k]["Canonical SMILES"] ==
        r["Canonical_SMILES"]
        and k in top10_exact_keys
        for k in top10_exact_keys
    )
]

top10_exact_rows.sort(
    key=lambda r: (
        r["Observed_Rank"],
        r["Canonical_SMILES"]
    )
)

# ============================================================
# 9. SAVE CSV FILES
# ============================================================

def write_dict_csv(path, rows):
    if not rows:
        return

    fields = list(rows[0].keys())

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields
        )
        w.writeheader()
        w.writerows(rows)

summary_path = os.path.join(
    OUTDIR,
    "VEGFR2_case_summary.csv"
)

seedwise_path = os.path.join(
    OUTDIR,
    "VEGFR2_case_seedwise.csv"
)

all_path = os.path.join(
    OUTDIR,
    "VEGFR2_case_all_compounds.csv"
)

top10_path = os.path.join(
    OUTDIR,
    "VEGFR2_case_observed_top10_ranks.csv"
)

write_dict_csv(
    summary_path,
    summary_rows
)

write_dict_csv(
    seedwise_path,
    seedwise_rows
)

write_dict_csv(
    all_path,
    all_rows
)

write_dict_csv(
    top10_path,
    top10_exact_rows
)

# ============================================================
# 10. MANIFEST / AUDIT RECORD
# ============================================================

manifest_path = os.path.join(
    OUTDIR,
    "VEGFR2_case_manifest.txt"
)

with open(
    manifest_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Exploratory post-hoc Davis cold-target case study\n"
    )
    f.write(
        f"Selected target: {TARGET}\n"
    )
    f.write(
        f"N frozen target records: {len(keys)}\n"
    )
    f.write(
        f"N unique compounds: "
        f"{len({metadata[k]['Canonical SMILES'] for k in keys})}\n"
    )
    f.write(
        f"Protein sequence length: {len(selected_seq)}\n"
    )
    f.write(
        "Target selection was locked before inspecting "
        "target-specific Prediction_raw values.\n"
    )
    f.write(
        "No retraining, hyperparameter tuning, checkpoint selection, "
        "or additional model selection was performed.\n"
    )
    f.write(
        "Predictions are the previously generated frozen held-out outputs.\n"
    )
    f.write(
        f"Target lock SHA256: {sha256_file(LOCK_FILE)}\n"
    )
    f.write(
        f"Raw label audit max abs difference: "
        f"{max_label_diff:.12g}\n"
    )
    f.write(
        "Higher frozen label values correspond to stronger affinity.\n"
    )

    f.write("\nINPUT FILE SHA256\n")

    for p in files:
        f.write(
            f"{sha256_file(p)}  {p}\n"
        )

# ============================================================
# 11. CONSOLE TOP-10 TABLE
# ============================================================

print()
print("=" * 90)
print("OBSERVED TOP-10 COMPOUNDS")
print("=" * 90)

for r in top10_exact_rows:
    ranks = " | ".join(
        f"{m}={r[m + '_Rank']}"
        for m in MODEL_ORDER
    )

    print(
        f"ObsRank={r['Observed_Rank']:<4} "
        f"Drug={r['Drug'][:24]:<24} "
        f"Label={float(r['Observed_Label']):.4f} | "
        f"{ranks}"
    )

print()
print("=" * 90)
print("SAVED OUTPUTS")
print("=" * 90)
print(summary_path)
print(seedwise_path)
print(all_path)
print(top10_path)
print(manifest_path)
