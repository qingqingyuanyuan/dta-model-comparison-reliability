#!/usr/bin/env python3
"""Synthetic V1C guard tests. No train/val/test dataset is loaded."""

from __future__ import annotations

import copy
import json
import tempfile
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from scripts import run_baseline_matrix as bm
from scripts import run_experiment_matrix as pm
from scripts import run_sensitivity_matrix as sm
from scripts.protocol_provenance import load_and_validate_plan, project_relative


def _write_primary_plan(path: Path):
    pm._write_plan(pm.build_full_matrix(), path)
    return load_and_validate_plan(
        path, cfg, expected_family="primary", expected_matrix=pm.build_full_matrix()
    )


def _base_ck(row, context, *, architecture, family, extra=None):
    payload = {
        "checkpoint_format_version": 5,
        "architecture_version": architecture,
        "dataset": row["dataset"],
        "setting": row["setting"],
        "model_seed": int(row["seed"]),
        "test_evaluated_during_training": False,
        "scaler": {"fit_on": "train_only"},
        "protocol_version": context["protocol_version"],
        "optimizer": context["optimizer"],
        "config_sha256": context["config_sha256"],
        "source_sha256": context["source_sha256"],
        "data_manifest_sha256": context["data_manifest_sha256"],
        "training_data_sha256": context["training_data_sha256"],
        "plan_sha256": context["plan_sha256"],
        "extra_metadata": {"experiment_family": family, **(extra or {})},
    }
    return payload


def main():
    assert cfg.TRAINING["optimizer"].lower() == "adamw"
    assert cfg.PROTOCOL_VERSION == "zhiy_graph_adamw_formal_v1"

    with tempfile.TemporaryDirectory(prefix="v1c_guard_") as td:
        td = Path(td)

        # Primary plan validation + frozen matrix tamper rejection.
        primary_plan = td / "primary.json"
        plan, pctx = _write_primary_plan(primary_plan)
        assert len(plan["matrix"]) == 54

        tampered = copy.deepcopy(plan)
        tampered["matrix"][0]["stem"] += "__tampered"
        bad_plan = td / "primary_tampered.json"
        bad_plan.write_text(json.dumps(tampered, indent=2), encoding="utf-8")
        try:
            load_and_validate_plan(
                bad_plan,
                cfg,
                expected_family="primary",
                expected_matrix=pm.build_full_matrix(),
            )
        except RuntimeError as exc:
            assert "FROZEN_PLAN_MISMATCH" in str(exc)
        else:
            raise AssertionError("tampered frozen matrix was not rejected")

        # Primary checkpoint: valid AdamW provenance accepted; legacy/Adam rejected.
        prow = copy.deepcopy(plan["matrix"][0])
        prow["checkpoint"] = project_relative(ROOT / "checkpoints" / "__v1c_test_primary.pt")
        ppath = ROOT / prow["checkpoint"]
        ppath.parent.mkdir(parents=True, exist_ok=True)
        good = _base_ck(
            prow,
            pctx,
            architecture=cfg.MODEL_ARCHITECTURE_VERSION,
            family="primary",
        )
        good["fusion_mode"] = prow["fusion"]
        torch.save(good, ppath)
        assert pm._checkpoint_matches(ppath, prow, pctx) is True

        legacy = copy.deepcopy(good)
        for key in ("protocol_version", "config_sha256", "source_sha256", "plan_sha256"):
            legacy.pop(key, None)
        legacy["optimizer"] = "adam"
        torch.save(legacy, ppath)
        assert pm._checkpoint_matches(ppath, prow, pctx) is False

        wrong_plan = copy.deepcopy(good)
        wrong_plan["plan_sha256"] = "0" * 64
        torch.save(wrong_plan, ppath)
        assert pm._checkpoint_matches(ppath, prow, pctx) is False

        wrong_data = copy.deepcopy(good)
        wrong_data["training_data_sha256"] = "0" * 64
        torch.save(wrong_data, ppath)
        assert pm._checkpoint_matches(ppath, prow, pctx) is False

        ppath.unlink(missing_ok=True)

        # Baseline matcher.
        baseline_plan = td / "baseline.json"
        brows = bm.build_matrix()
        bm._write_plan(brows, baseline_plan)
        bplan, bctx = load_and_validate_plan(
            baseline_plan,
            cfg,
            expected_family="controlled_baseline",
            expected_matrix=brows,
        )
        brow = copy.deepcopy(bplan["matrix"][0])
        brow["checkpoint"] = project_relative(ROOT / "checkpoints" / "baselines" / "__v1c_test_baseline.pt")
        bpath = ROOT / brow["checkpoint"]
        bpath.parent.mkdir(parents=True, exist_ok=True)
        goodb = _base_ck(
            brow,
            bctx,
            architecture=cfg.BASELINES[brow["baseline"]]["architecture_version"],
            family="controlled_baseline",
            extra={"baseline_name": brow["baseline"]},
        )
        torch.save(goodb, bpath)
        assert bm._checkpoint_matches(brow, bctx) is True
        badb = copy.deepcopy(goodb)
        badb["optimizer"] = "adam"
        torch.save(badb, bpath)
        assert bm._checkpoint_matches(brow, bctx) is False
        bpath.unlink(missing_ok=True)

        # Sensitivity matcher.
        sensitivity_plan = td / "sensitivity.json"
        srows = sm.build_matrix()
        sm._write_plan(srows, sensitivity_plan)
        splan, sctx = load_and_validate_plan(
            sensitivity_plan,
            cfg,
            expected_family="sensitivity",
            expected_matrix=srows,
        )
        srow = copy.deepcopy(splan["matrix"][0])
        srow["checkpoint"] = project_relative(ROOT / "checkpoints" / "sensitivity" / "__v1c_test_sensitivity.pt")
        spath = ROOT / srow["checkpoint"]
        spath.parent.mkdir(parents=True, exist_ok=True)
        goods = _base_ck(
            srow,
            sctx,
            architecture=cfg.MODEL_ARCHITECTURE_VERSION,
            family="sensitivity",
            extra={"sensitivity_name": srow["variant"]},
        )
        goods["fusion_mode"] = srow["fusion"]
        goods["config"] = {
            "PROTEIN_ENCODER": {"max_len": srow["overrides"]["protein_max_len"]},
            "DRUG_ENCODER": {"pooling": srow["overrides"]["drug_pooling"]},
        }
        torch.save(goods, spath)
        assert sm._sensitivity_ckpt_matches(srow, sctx) is True
        bads = copy.deepcopy(goods)
        bads["protocol_version"] = "old_protocol"
        torch.save(bads, spath)
        assert sm._sensitivity_ckpt_matches(srow, sctx) is False
        spath.unlink(missing_ok=True)

    print("V1C SYNTHETIC GUARDS = PASS")
    print("OLD ADAM / MISSING PROVENANCE REJECTION = PASS")
    print("VALID ADAMW PROVENANCE ACCEPTANCE = PASS")
    print("FROZEN MATRIX TAMPER STOP = PASS")
    print("HELD-OUT TEST = NOT LOADED")


if __name__ == "__main__":
    main()
