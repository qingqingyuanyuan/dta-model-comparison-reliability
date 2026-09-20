from pathlib import Path
import json

import pandas as pd
import torch

from data.utils import load_fixed_train_val, protein_seq_to_indices
from data.dataset import build_graph_cache, create_loader_from_frame
from models.colab_model import load_model


ROOT = Path.cwd()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = ROOT / "data_final"

N_PROTEINS = 16

WEIGHT_EPS = 1e-3
FEATURE_EPS = 1e-7
PRED_EPS = 1e-7

CASES = [
    {
        "name": "warm_concat_seed42",
        "setting": "warm",
        "ckpt": ROOT / "checkpoints" / "kiba_warm_concat_seed42__ADAMW_E50.pt",
    },
    {
        "name": "cold_drug_attention_seed42",
        "setting": "cold_drug",
        "ckpt": ROOT / "checkpoints" / "kiba_cold_drug_attention_seed42__ADAMW_E50.pt",
    },
    {
        "name": "cold_target_product_seed42",
        "setting": "cold_target",
        "ckpt": ROOT / "checkpoints" / "kiba_cold_target_product_seed42__ADAMW_E50.pt",
    },
]


def pick_encoded_unique_proteins(frame, max_len, truncation, n=16):
    seen = set()
    proteins = []

    for seq in dict.fromkeys(frame["Target Sequence"].astype(str).tolist()):
        encoded = protein_seq_to_indices(seq, int(max_len), truncation)
        key = tuple(encoded)

        if key in seen:
            continue

        seen.add(key)
        proteins.append(seq)

        if len(proteins) >= n:
            break

    if len(proteins) < 2:
        raise RuntimeError(
            f"Only {len(proteins)} unique encoded proteins available"
        )

    return proteins


def tensor_var(x):
    x = x.detach().float()
    if x.ndim < 2:
        return float(x.var(unbiased=False).item())
    return float(x.var(dim=0, unbiased=False).mean().item())


results = []

print("=" * 90)
print("ADAMW E50 FINAL REPRESENTATION / NORM GATE")
print("VALIDATION ONLY")
print("HELD-OUT TEST IS NOT LOADED")
print("DEVICE:", DEVICE)
print("=" * 90)

for case in CASES:
    print()
    print("=" * 90)
    print(case["name"])
    print("=" * 90)

    if not case["ckpt"].exists():
        raise FileNotFoundError(case["ckpt"])

    model, ckpt = load_model(case["ckpt"], allow_legacy=False)
    model = model.to(DEVICE).eval()

    pc = ckpt["config"]["PROTEIN_ENCODER"]

    max_len = int(pc["max_len"])
    truncation = pc.get("truncation", "right")

    _, val, _ = load_fixed_train_val(
        "kiba",
        case["setting"],
        data_root=DATA_ROOT,
    )

    proteins = pick_encoded_unique_proteins(
        val,
        max_len,
        truncation,
        N_PROTEINS,
    )

    fixed_drug = str(val["Canonical SMILES"].iloc[0])

    probe = pd.DataFrame(
        {
            "Canonical SMILES": [fixed_drug] * len(proteins),
            "Target Sequence": proteins,
            "Label_norm": [0.0] * len(proteins),
        }
    )

    graph_cache = build_graph_cache(
        [fixed_drug],
        show_progress=False,
    )

    loader = create_loader_from_frame(
        probe,
        graph_cache=graph_cache,
        batch_size=len(proteins),
        shuffle=False,
        model_seed=0,
        max_len=max_len,
        truncation=truncation,
        num_workers=0,
        trim_protein_padding=True,
        split_name=f"adamw_e50_final_gate/{case['name']}",
    )

    graph, protein, _ = next(iter(loader))

    graph = graph.to(DEVICE)
    protein = protein.to(DEVICE)

    with torch.no_grad():
        out = model(
            graph,
            protein,
            return_aux=True,
        )

    protein_global = out["protein_global"]
    prediction = out["prediction"].reshape(-1)

    enc = model.protein_encoder

    embedding_norm = float(
        enc.embedding.weight.detach().float().norm().item()
    )

    token_proj_norm = float(
        enc.token_proj.weight.detach().float().norm().item()
    )

    protein_global_var = tensor_var(protein_global)

    prediction_std = float(
        prediction.detach().float().std(unbiased=False).item()
    )

    embedding_max_abs = float(
        enc.embedding.weight.detach().float().abs().max().item()
    )

    token_proj_max_abs = float(
        enc.token_proj.weight.detach().float().abs().max().item()
    )

    passed = (
        embedding_norm > WEIGHT_EPS
        and token_proj_norm > WEIGHT_EPS
        and protein_global_var > FEATURE_EPS
        and prediction_std > PRED_EPS
    )

    row = {
        "name": case["name"],
        "checkpoint": str(case["ckpt"]),
        "n_unique_encoded_proteins": len(proteins),
        "embedding_weight_norm": embedding_norm,
        "token_proj_weight_norm": token_proj_norm,
        "embedding_max_abs": embedding_max_abs,
        "token_proj_max_abs": token_proj_max_abs,
        "protein_global_var": protein_global_var,
        "prediction_std_fixed_drug_changed_protein": prediction_std,
        "pass": passed,
    }

    results.append(row)

    print("n_unique_encoded_proteins =", len(proteins))
    print("embedding_weight_norm     =", f"{embedding_norm:.9e}")
    print("token_proj_weight_norm    =", f"{token_proj_norm:.9e}")
    print("embedding_max_abs         =", f"{embedding_max_abs:.9e}")
    print("token_proj_max_abs        =", f"{token_proj_max_abs:.9e}")
    print("protein_global_var        =", f"{protein_global_var:.9e}")
    print("prediction_std            =", f"{prediction_std:.9e}")
    print("FINAL_GATE                =", "PASS" if passed else "STOP")


out_dir = ROOT / "results" / "diagnostics" / "ADAMW_E50"
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "adamw_e50_final_gate.csv"
json_path = out_dir / "adamw_e50_final_gate.json"

pd.DataFrame(results).to_csv(csv_path, index=False)
json_path.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8",
)

print()
print("=" * 90)
print("FINAL SUMMARY")
print("=" * 90)

for x in results:
    print(
        f"{x['name']:35s} "
        f"=> {'PASS' if x['pass'] else 'STOP'}"
    )

n_pass = sum(int(x["pass"]) for x in results)

print()
print(f"PASS = {n_pass}/3")
print("CSV :", csv_path)
print("JSON:", json_path)
print("HELD-OUT TEST REMAINED SEALED")

if n_pass != 3:
    raise SystemExit(2)
