from __future__ import annotations

import copy
import shutil
import tempfile
import types
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train.trainer import Trainer


PROV = {
    "protocol_version": "v1d_synthetic_test",
    "optimizer": "adamw",
    "config_sha256": "a" * 64,
    "source_sha256": "b" * 64,
    "data_manifest_sha256": "e" * 64,
    "training_data_sha256": "f" * 64,
    "plan_sha256": "c" * 64,
}


CONFIG = {
    "PREDICTOR": {
        "task": "regression",
    },
    "TRAINING": {
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "scheduler": "reduce_on_plateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 10,
        "scheduler_min_lr": 1e-6,
        "num_epochs": 6,
        "early_stopping_patience": 30,
        "early_stopping_min_delta": 0.0,
        "gradient_clip_norm": 5.0,
        "seed": 777,
        "device": "cpu",
        "deterministic": True,

        # Do not intentionally trigger health STOPs
        # in this resume-equivalence test.
        "constant_prediction_start_epoch": 1000,
        "constant_prediction_patience": 3,
        "constant_prediction_std_eps": 1e-7,
        "branch_param_start_epoch": 1000,
        "branch_param_patience": 2,
        "branch_param_norm_eps": 1e-6,
    },
}


class TinyProtein(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(1, 4)
        self.token_proj = nn.Linear(4, 4)

    def forward(self, x):
        x = torch.relu(self.embedding(x))
        return torch.relu(self.token_proj(x))


class TinyDrug(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(1, 4)

    def forward(self, x):
        return torch.relu(self.input_proj(x))


class TinyModel(nn.Module):
    architecture_version = "v1d_resume_test"

    def __init__(self):
        super().__init__()
        self.protein_encoder = TinyProtein()
        self.drug_encoder = TinyDrug()
        self.head = nn.Linear(8, 1)

    def forward(self, drug, protein):
        d = self.drug_encoder(drug.float())
        p = self.protein_encoder(protein.float())
        return self.head(torch.cat([d, p], dim=1)).view(-1)


def make_base_state():
    torch.manual_seed(20260910)
    m = TinyModel()
    return copy.deepcopy(m.state_dict())


BASE_STATE = make_base_state()


def make_model():
    m = TinyModel()
    m.load_state_dict(BASE_STATE, strict=True)
    return m


def make_loaders():
    x1 = torch.linspace(-1.0, 1.0, 32).view(-1, 1)
    x2 = torch.linspace(1.0, -1.0, 32).view(-1, 1)
    y = (0.7 * x1[:, 0] - 0.4 * x2[:, 0] + 0.1)

    ds = TensorDataset(x1, x2, y)

    train_loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def make_trainer():
    return Trainer(
        make_model(),
        copy.deepcopy(CONFIG),
        device="cpu",
    )


def assert_histories_equal(a, b):
    keys = [
        "epoch",
        "train_loss",
        "val_loss",
        "lr",
        "val_prediction_std",
        "total_grad_norm",
        "protein_grad_norm",
        "drug_grad_norm",
        "protein_param_norm",
        "drug_param_norm",
        "protein_embedding_norm",
        "protein_token_proj_norm",
        "drug_input_proj_norm",
    ]

    for key in keys:
        aa = np.asarray(a[key])
        bb = np.asarray(b[key])

        if aa.shape != bb.shape:
            raise AssertionError(
                f"history shape mismatch: {key}"
            )

        if np.issubdtype(aa.dtype, np.number):
            if not np.array_equal(aa, bb):
                raise AssertionError(
                    f"history mismatch: {key}\n"
                    f"{aa}\n{bb}"
                )
        elif aa.tolist() != bb.tolist():
            raise AssertionError(
                f"history mismatch: {key}"
            )


with tempfile.TemporaryDirectory(
    prefix="v1d_resume_"
) as td:

    td = Path(td)

    # ========================================================
    # 1. Uninterrupted reference
    # ========================================================

    full = make_trainer()
    full_train, full_val = make_loaders()

    full_history = full.fit(
        full_train,
        full_val,
    )

    full_state = {
        k: v.detach().cpu().clone()
        for k, v in full.model.state_dict().items()
    }

    print("UNINTERRUPTED REFERENCE = PASS")

    # ========================================================
    # 2. Intentional infrastructure-like interruption
    #    immediately AFTER epoch-2 LAST_STATE commit.
    # ========================================================

    last_state = td / "tiny.pt.last_state"

    partial = make_trainer()
    part_train, part_val = make_loaders()

    original_save = partial._save_last_state

    def hooked_save(
        self,
        path,
        epoch,
        protocol_metadata,
        resume_key,
    ):
        original_save(
            path,
            epoch,
            protocol_metadata,
            resume_key,
        )

        if int(epoch) == 2:
            raise KeyboardInterrupt(
                "intentional V1D resume test interruption"
            )

    partial._save_last_state = types.MethodType(
        hooked_save,
        partial,
    )

    interrupted = False

    try:
        partial.fit(
            part_train,
            part_val,
            last_state_path=last_state,
            protocol_metadata=PROV,
            resume_key="tiny_run",
        )
    except KeyboardInterrupt:
        interrupted = True

    if not interrupted:
        raise AssertionError(
            "intentional interruption did not occur"
        )

    if not last_state.exists():
        raise AssertionError(
            "LAST_STATE was not created"
        )

    print("INTERRUPTION AFTER ATOMIC LAST_STATE = PASS")

    # Preserve epoch-2 copy for rejection tests.
    rejection_state = td / "reject_test.pt"
    shutil.copy2(last_state, rejection_state)

    # ========================================================
    # 3. Provenance mismatch must STOP
    # ========================================================

    bad_prov = dict(PROV)
    bad_prov["plan_sha256"] = "d" * 64

    bad = make_trainer()
    bad_train, bad_val = make_loaders()

    rejected = False

    try:
        bad.fit(
            bad_train,
            bad_val,
            last_state_path=rejection_state,
            protocol_metadata=bad_prov,
            resume_key="tiny_run",
        )
    except RuntimeError as exc:
        if "RESUME_PROVENANCE_MISMATCH" in str(exc):
            rejected = True

    if not rejected:
        raise AssertionError(
            "provenance-mismatched LAST_STATE was not rejected"
        )

    print("RESUME PROVENANCE MISMATCH REJECT = PASS")


    # ========================================================
    # 3b. Training-data provenance mismatch must also STOP
    # ========================================================

    bad_data_prov = dict(PROV)
    bad_data_prov["training_data_sha256"] = "0" * 64

    bad_data = make_trainer()
    bad_data_train, bad_data_val = make_loaders()

    data_rejected = False

    try:
        bad_data.fit(
            bad_data_train,
            bad_data_val,
            last_state_path=rejection_state,
            protocol_metadata=bad_data_prov,
            resume_key="tiny_run",
        )
    except RuntimeError as exc:
        if "RESUME_PROVENANCE_MISMATCH" in str(exc):
            data_rejected = True

    if not data_rejected:
        raise AssertionError(
            "training-data-mismatched LAST_STATE was not rejected"
        )

    print("RESUME DATA PROVENANCE MISMATCH REJECT = PASS")

    # ========================================================
    # 4. Run identity mismatch must STOP
    # ========================================================

    bad_key = make_trainer()
    key_train, key_val = make_loaders()

    rejected = False

    try:
        bad_key.fit(
            key_train,
            key_val,
            last_state_path=rejection_state,
            protocol_metadata=PROV,
            resume_key="WRONG_RUN",
        )
    except RuntimeError as exc:
        if "RESUME_RUN_IDENTITY_MISMATCH" in str(exc):
            rejected = True

    if not rejected:
        raise AssertionError(
            "run-identity-mismatched LAST_STATE was not rejected"
        )

    print("RESUME RUN IDENTITY REJECT = PASS")

    # ========================================================
    # 5. Correct resume
    # ========================================================

    resumed = make_trainer()
    resume_train, resume_val = make_loaders()

    resumed_history = resumed.fit(
        resume_train,
        resume_val,
        last_state_path=last_state,
        protocol_metadata=PROV,
        resume_key="tiny_run",
    )

    if resumed_history["epoch"] != [1, 2, 3, 4, 5, 6]:
        raise AssertionError(
            f"bad resumed epochs: "
            f"{resumed_history['epoch']}"
        )

    assert_histories_equal(
        full_history,
        resumed_history,
    )

    for key, expected in full_state.items():
        actual = resumed.model.state_dict()[key].detach().cpu()
        if not torch.equal(expected, actual):
            raise AssertionError(
                f"final model mismatch: {key}"
            )

    if full.best_epoch != resumed.best_epoch:
        raise AssertionError(
            "best_epoch mismatch"
        )

    if full.best_val_loss != resumed.best_val_loss:
        raise AssertionError(
            "best_val_loss mismatch"
        )

    print("INTERRUPTED + RESUMED == UNINTERRUPTED = PASS")


print()
print("===============================================")
print("V1D RESUME SYNTHETIC SUMMARY")
print("===============================================")
print("LAST_STATE atomic checkpoint         = PASS")
print("epoch continuation                   = PASS")
print("model state restoration              = PASS")
print("optimizer state restoration          = PASS")
print("scheduler state restoration          = PASS")
print("history continuation                 = PASS")
print("early/health counters serialized     = PASS")
print("RNG restoration                      = PASS")
print("provenance mismatch rejection        = PASS")
print("run identity mismatch rejection      = PASS")
print("trajectory equivalence               = PASS")
print("HELD-OUT TEST                        = NOT LOADED")
print("FORMAL 126                           = NOT STARTED")
print("===============================================")
