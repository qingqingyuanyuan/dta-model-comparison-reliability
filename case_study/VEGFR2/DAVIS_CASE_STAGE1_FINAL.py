import csv
import json
import glob
import hashlib
from collections import defaultdict, Counter

BASE = "./FINAL_TEST_WORKSPACE/evaluation_outputs"

REFERENCE = (
    BASE +
    "/davis_cold_target_product_seed123__davis__cold_target__test_predictions.csv"
)

TARGETS = "./FINAL_TEST_WORKSPACE/data/raw/davis/targets.txt"
OUT = "davis_cold_target_FINAL_candidate_targets.csv"

# ---------------------------------------------------
# helpers
# ---------------------------------------------------

def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def seqkey(seq):
    return hashlib.sha256(seq.encode("utf-8")).hexdigest()[:16]

def frozen_key(r):
    return (
        r["Target Sequence"],
        r["Canonical SMILES"],
        r["Label"],
        r["Source Split"],
        r["Source Row"],
    )

# ---------------------------------------------------
# 1. Find the 15 formal prediction files
# ---------------------------------------------------

files = sorted(
    p for p in glob.glob(
        BASE + "/**/*davis*cold_target*test_predictions.csv",
        recursive=True
    )
    if "/sensitivity/" not in p
)

print("=" * 90)
print("FORMAL DAVIS COLD-TARGET PREDICTION FILES")
print("=" * 90)
print("N files:", len(files))

if len(files) != 15:
    raise RuntimeError(
        f"Expected 15 formal files (5 models x 3 seeds), found {len(files)}"
    )

# ---------------------------------------------------
# 2. Verify all 15 use exactly the same frozen test records
# ---------------------------------------------------

ref_rows = read_csv(REFERENCE)
ref_counter = Counter(frozen_key(r) for r in ref_rows)

print("Reference frozen test rows:", len(ref_rows))

all_same = True

for p in files:
    rows = read_csv(p)
    c = Counter(frozen_key(r) for r in rows)
    same = (c == ref_counter)

    print(
        ("OK   " if same else "FAIL "),
        len(rows),
        p
    )

    if not same:
        all_same = False

print()
print("All 15 files contain identical frozen test records:", all_same)

if not all_same:
    raise RuntimeError(
        "Formal prediction files do not share an identical frozen test set."
    )

# ---------------------------------------------------
# 3. Target-name mapping
# ---------------------------------------------------

with open(TARGETS, "r", encoding="utf-8") as f:
    target_map = json.load(f)

seq_to_reference_names = defaultdict(list)

for name, seq in target_map.items():
    seq_to_reference_names[seq].append(name)

# ---------------------------------------------------
# 4. Build target candidates using FINAL frozen test only
#    Prediction values are NOT used.
# ---------------------------------------------------

groups = defaultdict(list)

for r in ref_rows:
    groups[r["Target Sequence"]].append(r)

records = []

for seq, rows in groups.items():

    ids = sorted({r["Target ID"] for r in rows})
    names = sorted(seq_to_reference_names.get(seq, []))
    drugs = {r["Canonical SMILES"] for r in rows}

    labels = [float(r["Label"]) for r in rows]

    records.append({
        "SequenceKey": seqkey(seq),
        "N_records": len(rows),
        "N_unique_drugs": len(drugs),
        "N_test_target_IDs": len(ids),
        "Test_target_IDs": " | ".join(ids),
        "Reference_target_names": " | ".join(names),
        "N_reference_names_same_sequence": len(names),
        "Label_min": min(labels),
        "Label_max": max(labels),
        "Sequence_length": len(seq),
        "Clean_single_target": (
            "YES"
            if len(ids) == 1 and len(names) == 1
            else "NO"
        ),
    })

records.sort(
    key=lambda x: (
        x["Clean_single_target"] != "YES",
        -x["N_unique_drugs"],
        x["Test_target_IDs"],
    )
)

fields = [
    "SequenceKey",
    "N_records",
    "N_unique_drugs",
    "N_test_target_IDs",
    "Test_target_IDs",
    "Reference_target_names",
    "N_reference_names_same_sequence",
    "Label_min",
    "Label_max",
    "Sequence_length",
    "Clean_single_target",
]

with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(records)

clean = [r for r in records if r["Clean_single_target"] == "YES"]

print()
print("=" * 90)
print("FINAL FROZEN TEST SUMMARY")
print("=" * 90)
print("rows:", len(ref_rows))
print("unique protein sequences:", len(groups))
print("clean single-target candidates:", len(clean))

print()
print("=" * 90)
print("CLEAN FINAL CANDIDATES")
print("No model prediction value was used for this list.")
print("=" * 90)

for r in clean:
    print(
        f'{r["Test_target_IDs"]:<34} '
        f'drugs={r["N_unique_drugs"]:<3} '
        f'rows={r["N_records"]:<3} '
        f'len={r["Sequence_length"]:<5} '
        f'label=[{r["Label_min"]:.3f}, {r["Label_max"]:.3f}]'
    )

print()
print("Saved:", OUT)
