from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.predictor import DTAPredictor
from scripts.runtime_health_gate import run_canonical_health_gate
from train.trainer import Trainer


ROOT = Path(__file__).resolve().parents[1]

CKPT = (
    ROOT
    / "checkpoints"
    / "kiba_warm_concat_seed42__HARDEN_GRAD_SMOKE2.pt"
)

VAL_FILE = (
    ROOT
    / "data_final"
    / "KIBA"
    / "warm"
    / "val.csv"
)

VAL_PRED_FILE = (
    ROOT
    / "training_outputs"
    / "kiba_warm_concat_seed42__HARDEN_GRAD_SMOKE2__val_predictions.csv"
)

OUTDIR = (
    ROOT
    / "results"
    / "preformal_hardening"
    / "negative_controls"
)


def trusted_load(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def load_predictions(path):
    df = pd.read_csv(path)

    preferred = [
        "Prediction_norm",
        "prediction_norm",
        "pred_norm",
        "prediction",
        "pred",
        "y_pred",
    ]

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    col = None

    for name in preferred:
        if name.lower() in lower:
            col = lower[name.lower()]
            break

    if col is None:
        for c in df.columns:
            if "pred" in str(c).lower():
                col = c
                break

    if col is None:
        raise RuntimeError(
            f"No prediction column found. Columns={list(df.columns)}"
        )

    x = df[col].to_numpy(dtype=float)

    if not np.isfinite(x).all():
        raise RuntimeError(
            "Healthy validation predictions already contain NaN/Inf"
        )

    return x


def fresh_model(ck, device):
    conf = copy.deepcopy(ck["config"])

    model = DTAPredictor(conf)

    model.load_state_dict(
        ck["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model, conf


def expect_health_stop(
    *,
    name,
    model,
    val_df,
    conf,
    val_pred,
    expected_failure,
):
    path = OUTDIR / f"{name}.json"

    try:
        run_canonical_health_gate(
            model,
            val_df,
            conf,
            val_pred,
            dataset="KIBA",
            setting="warm",
            stem=name,
            output_path=path,
        )

    except RuntimeError as exc:
        if not path.exists():
            raise RuntimeError(
                f"{name}: gate raised but JSON was not written"
            ) from exc

        result = json.loads(
            path.read_text(encoding="utf-8")
        )

        failures = result.get("failures") or []

        if result.get("status") != "STOP":
            raise RuntimeError(
                f"{name}: expected status STOP, got {result.get('status')}"
            )

        if expected_failure not in failures:
            raise RuntimeError(
                f"{name}: expected {expected_failure}; "
                f"got failures={failures}"
            )

        print(
            f"[PASS] {name}: "
            f"caught {expected_failure}"
        )

        return

    raise RuntimeError(
        f"{name}: BAD MODEL WAS NOT STOPPED"
    )


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not CKPT.exists():
        raise FileNotFoundError(CKPT)

    if not VAL_FILE.exists():
        raise FileNotFoundError(VAL_FILE)

    if not VAL_PRED_FILE.exists():
        raise FileNotFoundError(VAL_PRED_FILE)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print("PRE-FORMAL NEGATIVE CONTROL")
    print("VALIDATION ONLY")
    print("HELD-OUT TEST IS NOT LOADED")
    print("DEVICE:", device)
    print("=" * 78)

    ck = trusted_load(CKPT)

    if (
        ck.get("test_evaluated_during_training")
        is not False
    ):
        raise RuntimeError(
            "Source checkpoint test seal invalid"
        )

    val_df = pd.read_csv(VAL_FILE)

    healthy_pred = load_predictions(
        VAL_PRED_FILE
    )

    if len(healthy_pred) != len(val_df):
        raise RuntimeError(
            "Validation prediction length mismatch"
        )

    passed = 0

    # =========================================================
    # TEST 1 — constant final prediction
    # =========================================================
    model, conf = fresh_model(
        ck,
        device,
    )

    constant_pred = np.full_like(
        healthy_pred,
        float(np.mean(healthy_pred)),
        dtype=float,
    )

    expect_health_stop(
        name="NEG_CONSTANT_OUTPUT",
        model=model,
        val_df=val_df,
        conf=conf,
        val_pred=constant_pred,
        expected_failure=(
            "CONSTANT_OUTPUT_COLLAPSE"
        ),
    )

    passed += 1

    del model

    # =========================================================
    # TEST 2 — destroy protein branch in memory
    # =========================================================
    model, conf = fresh_model(
        ck,
        device,
    )

    with torch.no_grad():
        for p in model.protein_encoder.parameters():
            p.zero_()

    expect_health_stop(
        name="NEG_PROTEIN_ZERO",
        model=model,
        val_df=val_df,
        conf=conf,
        val_pred=healthy_pred,
        expected_failure=(
            "PROTEIN_PARAMETER_COLLAPSE"
        ),
    )

    passed += 1

    del model

    # =========================================================
    # TEST 3 — destroy drug branch in memory
    # =========================================================
    model, conf = fresh_model(
        ck,
        device,
    )

    with torch.no_grad():
        for p in model.drug_encoder.parameters():
            p.zero_()

    expect_health_stop(
        name="NEG_DRUG_ZERO",
        model=model,
        val_df=val_df,
        conf=conf,
        val_pred=healthy_pred,
        expected_failure=(
            "DRUG_PARAMETER_COLLAPSE"
        ),
    )

    passed += 1

    del model

    # =========================================================
    # TEST 4 — NaN must remain fatal
    # Test both parameter finite and gradient finite guards.
    # =========================================================
    model, conf = fresh_model(
        ck,
        torch.device("cpu"),
    )

    trainer = Trainer(
        model,
        conf,
        device="cpu",
    )

    # Parameter NaN
    first_param = next(
        model.parameters()
    )

    original_value = (
        first_param.view(-1)[0]
        .detach()
        .clone()
    )

    with torch.no_grad():
        first_param.view(-1)[0] = float("nan")

    try:
        trainer._assert_parameters_finite()
    except FloatingPointError:
        print(
            "[PASS] NEG_NONFINITE_PARAMETER: "
            "FloatingPointError caught"
        )
    else:
        raise RuntimeError(
            "NEG_NONFINITE_PARAMETER WAS NOT STOPPED"
        )

    # Restore parameter
    with torch.no_grad():
        first_param.view(-1)[0] = original_value

    # Gradient NaN
    protein_param = next(
        model.protein_encoder.parameters()
    )

    protein_param.grad = torch.zeros_like(
        protein_param
    )

    protein_param.grad.view(-1)[0] = float("nan")

    try:
        trainer._module_grad_norm(
            model.protein_encoder
        )
    except FloatingPointError:
        print(
            "[PASS] NEG_NONFINITE_GRADIENT: "
            "FloatingPointError caught"
        )
    else:
        raise RuntimeError(
            "NEG_NONFINITE_GRADIENT WAS NOT STOPPED"
        )

    passed += 1

    print()
    print("=" * 78)
    print(f"NEGATIVE CONTROL SUMMARY = {passed}/4 PASS")
    print("NORMAL MODEL              = previously PASS")
    print("BAD MODELS                = correctly rejected")
    print("HELD-OUT TEST             = SEALED")
    print("=" * 78)


if __name__ == "__main__":
    main()
