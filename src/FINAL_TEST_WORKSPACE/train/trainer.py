"""Training engine for the sealed-test ZhiYao-Graph experiment protocol."""

from __future__ import annotations

import copy
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    """Train on fixed train/validation splits and select by validation MSE only.

    The held-out test split is intentionally outside this class. Final test
    evaluation is performed later by ``scripts/evaluate_model.py`` after the
    experiment matrix has been fixed/trained.
    """

    def __init__(self, model, config, device=None):
        self.model = model
        self.config = copy.deepcopy(config)
        tc = self.config["TRAINING"]

        if str(self.config["PREDICTOR"].get("task", "regression")).lower() != "regression":
            raise RuntimeError("Final Trainer supports continuous DTA regression only")

        self.model_seed = int(tc.get("seed", 42))
        self._seed_runtime(self.model_seed, bool(tc.get("deterministic", True)))

        requested_device = str(tc.get("device", "auto"))
        if device is not None:
            self.device = torch.device(device)
        elif requested_device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(requested_device)

        self.model = self.model.to(self.device)
        self.criterion = nn.MSELoss(reduction="mean")

        optimizer_name = str(tc.get("optimizer", "adamw")).lower()
        if optimizer_name == "adam":
            optimizer_cls = torch.optim.Adam
        elif optimizer_name == "adamw":
            optimizer_cls = torch.optim.AdamW
        else:
            raise ValueError(
                f"Unsupported optimizer: {optimizer_name}"
            )

        self.optimizer = optimizer_cls(
            self.model.parameters(),
            lr=float(tc["learning_rate"]),
            weight_decay=float(tc["weight_decay"]),
        )

        scheduler_name = str(tc.get("scheduler", "reduce_on_plateau")).lower()
        if scheduler_name != "reduce_on_plateau":
            raise ValueError(f"Unsupported scheduler for final protocol: {scheduler_name}")
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=float(tc.get("scheduler_factor", 0.5)),
            patience=int(tc.get("scheduler_patience", 10)),
            min_lr=float(tc.get("scheduler_min_lr", 1e-6)),
        )

        self.num_epochs = int(tc["num_epochs"])
        self.patience = int(tc.get("early_stopping_patience", tc.get("patience", 30)))
        self.min_delta = float(tc.get("early_stopping_min_delta", 0.0))
        self.gradient_clip_norm = float(tc.get("gradient_clip_norm", 5.0))
        if self.num_epochs <= 0 or self.patience <= 0:
            raise ValueError("num_epochs and early_stopping_patience must be positive")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")

        self.best_val_loss = float("inf")
        self.best_epoch: Optional[int] = None
        self.stop_epoch: Optional[int] = None
        self.stop_reason = "not_started"
        self.patience_counter = 0

        self.best_model_state = None
        self.best_optimizer_state = None
        self.best_scheduler_state = None

        # Pre-formal numerical / collapse guards.
        # These are engineering safety gates; they do not change model math.
        self.constant_prediction_std_eps = float(
            tc.get("constant_prediction_std_eps", 1e-7)
        )
        self.constant_prediction_start_epoch = int(
            tc.get("constant_prediction_start_epoch", 10)
        )
        self.constant_prediction_patience = int(
            tc.get("constant_prediction_patience", 3)
        )
        self.branch_param_norm_eps = float(
            tc.get("branch_param_norm_eps", 1e-6)
        )
        self.branch_param_start_epoch = int(
            tc.get("branch_param_start_epoch", 5)
        )
        self.branch_param_patience = int(
            tc.get("branch_param_patience", 2)
        )

        self._constant_prediction_streak = 0
        self._protein_param_streak = 0
        self._drug_param_streak = 0
        self._epoch_grad_snapshot: Dict[str, float] = {}

        self.history: Dict[str, Any] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "val_prediction_std": [],
            "total_grad_norm": [],
            "protein_grad_norm": [],
            "drug_grad_norm": [],
            "protein_param_norm": [],
            "drug_param_norm": [],
            "protein_embedding_norm": [],
            "protein_token_proj_norm": [],
            "drug_input_proj_norm": [],
        }

        print(f"Device: {self.device}")
        print(f"Model seed: {self.model_seed}")
        print(
            "Early stopping: "
            f"patience={self.patience}, min_delta={self.min_delta:g}; "
            f"max_epochs={self.num_epochs}"
        )

    @staticmethod
    def _seed_runtime(seed: int, deterministic: bool) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = bool(deterministic)
            torch.backends.cudnn.benchmark = False if deterministic else False
        # PyG scatter reductions on GPU can remain nondeterministic on some
        # hardware/PyTorch combinations, so we do not force
        # torch.use_deterministic_algorithms(True), which can make valid GNN
        # kernels fail. Seeds + deterministic cuDNN settings are recorded.

    @staticmethod
    def _batch_size(affinities: torch.Tensor) -> int:
        return int(affinities.numel())

    @staticmethod
    def _module_param_norm(module) -> float:
        if module is None:
            return float("nan")
        total = 0.0
        found = False
        with torch.no_grad():
            for param in module.parameters():
                found = True
                total += float(
                    param.detach().float().pow(2).sum().item()
                )
        return total ** 0.5 if found else float("nan")

    @staticmethod
    def _module_grad_norm(module) -> float:
        if module is None:
            return float("nan")
        total = 0.0
        found = False
        for param in module.parameters():
            if param.grad is None:
                continue
            found = True
            g = param.grad.detach().float()
            if not torch.isfinite(g).all():
                raise FloatingPointError(
                    "Non-finite gradient detected in model branch"
                )
            total += float(g.pow(2).sum().item())
        return total ** 0.5 if found else 0.0

    @staticmethod
    def _weight_norm(module, attr_name: str) -> float:
        if module is None or not hasattr(module, attr_name):
            return float("nan")
        obj = getattr(module, attr_name)
        weight = getattr(obj, "weight", None)
        if weight is None:
            return float("nan")
        return float(weight.detach().float().norm().item())

    def _parameter_snapshot(self) -> Dict[str, float]:
        protein = getattr(self.model, "protein_encoder", None)
        drug = getattr(self.model, "drug_encoder", None)

        return {
            "protein_param_norm": self._module_param_norm(protein),
            "drug_param_norm": self._module_param_norm(drug),
            "protein_embedding_norm": self._weight_norm(
                protein, "embedding"
            ),
            "protein_token_proj_norm": self._weight_norm(
                protein, "token_proj"
            ),
            "drug_input_proj_norm": self._weight_norm(
                drug, "input_proj"
            ),
        }

    def _assert_parameters_finite(self) -> None:
        bad = []
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not torch.isfinite(param).all():
                    bad.append(name)
        if bad:
            raise FloatingPointError(
                "Non-finite model parameters after optimizer step: "
                + ", ".join(bad[:10])
            )

    def train_epoch(self, train_loader) -> float:
        self.model.train()
        total_squared_error_mean = 0.0
        total_samples = 0
        self._epoch_grad_snapshot = {}

        if len(train_loader) == 0:
            raise ValueError("train_loader is empty")

        for drug_graphs, protein_seqs, affinities in tqdm(train_loader, desc="train"):
            drug_graphs = drug_graphs.to(self.device, non_blocking=True)
            protein_seqs = protein_seqs.to(self.device, non_blocking=True)
            affinities = affinities.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(drug_graphs, protein_seqs)

            if not torch.isfinite(pred).all():
                raise FloatingPointError(
                    "Non-finite training prediction detected"
                )

            loss = self.criterion(pred, affinities)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss: {loss.item()}"
                )

            loss.backward()

            # Record all gradient-flow diagnostics from the SAME first
            # real batch of each epoch so total/branch norms are comparable.
            capture_grad_snapshot = not self._epoch_grad_snapshot

            if capture_grad_snapshot:
                self._epoch_grad_snapshot = {
                    "protein_grad_norm": self._module_grad_norm(
                        getattr(self.model, "protein_encoder", None)
                    ),
                    "drug_grad_norm": self._module_grad_norm(
                        getattr(self.model, "drug_encoder", None)
                    ),
                }

            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.gradient_clip_norm,
                error_if_nonfinite=True,
            )

            if not torch.isfinite(total_grad_norm):
                raise FloatingPointError(
                    "Non-finite total gradient norm detected"
                )

            if capture_grad_snapshot:
                self._epoch_grad_snapshot["total_grad_norm"] = float(
                    total_grad_norm.detach().cpu().item()
                )

            self.optimizer.step()
            self._assert_parameters_finite()

            bs = self._batch_size(affinities)
            total_squared_error_mean += float(loss.item()) * bs
            total_samples += bs

        if total_samples == 0:
            raise RuntimeError("No training samples were processed")
        return total_squared_error_mean / total_samples

    @torch.no_grad()
    def evaluate(self, loader):
        """Return sample-weighted MSE, predictions and labels for train/val use."""
        self.model.eval()
        if len(loader) == 0:
            raise ValueError("evaluation loader is empty")

        weighted_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []

        for drug_graphs, protein_seqs, affinities in loader:
            drug_graphs = drug_graphs.to(self.device, non_blocking=True)
            protein_seqs = protein_seqs.to(self.device, non_blocking=True)
            affinities = affinities.to(self.device, non_blocking=True)

            pred = self.model(drug_graphs, protein_seqs)

            if not torch.isfinite(pred).all():
                raise FloatingPointError(
                    "Non-finite evaluation prediction detected"
                )

            loss = self.criterion(pred, affinities)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite evaluation loss: {loss.item()}")

            bs = self._batch_size(affinities)
            weighted_loss += float(loss.item()) * bs
            total_samples += bs
            all_preds.append(pred.detach().cpu().numpy().reshape(-1))
            all_labels.append(affinities.detach().cpu().numpy().reshape(-1))

        if total_samples == 0:
            raise RuntimeError("No evaluation samples were processed")
        return (
            weighted_loss / total_samples,
            np.concatenate(all_preds, axis=0),
            np.concatenate(all_labels, axis=0),
        )

    def _is_improvement(self, val_loss: float) -> bool:
        return val_loss < (self.best_val_loss - self.min_delta)

    def _capture_best_state(self, epoch: int, val_loss: float) -> None:
        self.best_val_loss = float(val_loss)
        self.best_epoch = int(epoch)
        self.best_model_state = {
            k: v.detach().cpu().clone()
            for k, v in self.model.state_dict().items()
        }
        # Keep optimizer/scheduler states matched to the selected model state.
        self.best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self.best_scheduler_state = copy.deepcopy(self.scheduler.state_dict())

    @staticmethod
    def _resume_provenance(protocol_metadata):
        if protocol_metadata is None:
            raise RuntimeError(
                "RESUME_PROVENANCE_MISSING — STOP: "
                "LAST_STATE requires protocol metadata"
            )

        required = (
            "protocol_version",
            "optimizer",
            "config_sha256",
            "source_sha256",
            "data_manifest_sha256",
            "training_data_sha256",
            "plan_sha256",
        )

        missing = [
            key for key in required
            if not protocol_metadata.get(key)
        ]
        if missing:
            raise RuntimeError(
                "RESUME_PROVENANCE_MISSING — STOP: "
                + ", ".join(missing)
            )

        out = {
            key: str(protocol_metadata[key])
            for key in required
        }
        out["optimizer"] = out["optimizer"].lower()
        return out

    def _capture_rng_state(self):
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().cpu(),
            "cuda_used": self.device.type == "cuda",
            "torch_cuda_all": None,
        }

        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "RESUME_RNG_FAILURE — STOP: "
                    "Trainer device is CUDA but CUDA is unavailable"
                )
            state["torch_cuda_all"] = [
                x.cpu()
                for x in torch.cuda.get_rng_state_all()
            ]

        return state

    def _restore_rng_state(self, state):
        if not isinstance(state, dict):
            raise RuntimeError(
                "RESUME_RNG_STATE_INVALID — STOP"
            )

        required = (
            "python",
            "numpy",
            "torch_cpu",
            "cuda_used",
            "torch_cuda_all",
        )
        missing = [k for k in required if k not in state]
        if missing:
            raise RuntimeError(
                "RESUME_RNG_STATE_INVALID — STOP: "
                + ", ".join(missing)
            )

        saved_cuda = bool(state["cuda_used"])
        current_cuda = self.device.type == "cuda"

        if saved_cuda != current_cuda:
            raise RuntimeError(
                "RESUME_DEVICE_MISMATCH — STOP: "
                f"saved_cuda={saved_cuda}, current_cuda={current_cuda}"
            )

        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"].cpu())

        if saved_cuda:
            cuda_states = state["torch_cuda_all"]
            if not isinstance(cuda_states, (list, tuple)):
                raise RuntimeError(
                    "RESUME_CUDA_RNG_INVALID — STOP"
                )

            if len(cuda_states) != torch.cuda.device_count():
                raise RuntimeError(
                    "RESUME_CUDA_DEVICE_COUNT_MISMATCH — STOP: "
                    f"saved={len(cuda_states)}, "
                    f"current={torch.cuda.device_count()}"
                )

            torch.cuda.set_rng_state_all(
                [x.cpu() for x in cuda_states]
            )

    def _save_last_state(
        self,
        path,
        epoch,
        protocol_metadata,
        resume_key,
    ):
        path = os.fspath(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        provenance = self._resume_provenance(
            protocol_metadata
        )

        configured_optimizer = str(
            self.config.get("TRAINING", {}).get(
                "optimizer", ""
            )
        ).lower()

        if provenance["optimizer"] != configured_optimizer:
            raise RuntimeError(
                "RESUME_OPTIMIZER_MISMATCH — STOP"
            )

        state = {
            "last_state_format_version": 1,
            "state_kind": "last_completed_epoch",
            "last_completed_epoch": int(epoch),
            "resume_key": str(resume_key),
            "model_seed": int(self.model_seed),

            "model_state_dict": {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
            },
            "optimizer_state_dict": copy.deepcopy(
                self.optimizer.state_dict()
            ),
            "scheduler_state_dict": copy.deepcopy(
                self.scheduler.state_dict()
            ),

            "patience_counter": int(
                self.patience_counter
            ),

            "constant_prediction_streak": int(
                self._constant_prediction_streak
            ),
            "protein_param_streak": int(
                self._protein_param_streak
            ),
            "drug_param_streak": int(
                self._drug_param_streak
            ),

            "best_val_loss": float(
                self.best_val_loss
            ),
            "best_epoch": self.best_epoch,
            "best_model_state": copy.deepcopy(
                self.best_model_state
            ),
            "best_optimizer_state": copy.deepcopy(
                self.best_optimizer_state
            ),
            "best_scheduler_state": copy.deepcopy(
                self.best_scheduler_state
            ),

            "stop_epoch": self.stop_epoch,
            "stop_reason": self.stop_reason,

            "history": copy.deepcopy(self.history),
            "rng_state": self._capture_rng_state(),

            "protocol_provenance": provenance,

            "test_evaluated_during_training": False,
        }

        tmp_path = path + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)

    @staticmethod
    def _trusted_resume_load(path):
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

    def _load_last_state(
        self,
        path,
        protocol_metadata,
        resume_key,
    ):
        path = os.fspath(path)

        state = self._trusted_resume_load(path)

        if not isinstance(state, dict):
            raise RuntimeError(
                "RESUME_STATE_INVALID — STOP"
            )

        if state.get("last_state_format_version") != 1:
            raise RuntimeError(
                "RESUME_FORMAT_MISMATCH — STOP"
            )

        if state.get("state_kind") != "last_completed_epoch":
            raise RuntimeError(
                "RESUME_STATE_KIND_MISMATCH — STOP"
            )

        expected_provenance = self._resume_provenance(
            protocol_metadata
        )
        saved_provenance = state.get(
            "protocol_provenance"
        )

        if saved_provenance != expected_provenance:
            raise RuntimeError(
                "RESUME_PROVENANCE_MISMATCH — STOP"
            )

        if str(state.get("resume_key")) != str(resume_key):
            raise RuntimeError(
                "RESUME_RUN_IDENTITY_MISMATCH — STOP"
            )

        if int(state.get("model_seed", -1)) != int(
            self.model_seed
        ):
            raise RuntimeError(
                "RESUME_SEED_MISMATCH — STOP"
            )

        epoch = int(
            state.get("last_completed_epoch", -1)
        )

        if epoch < 1 or epoch > self.num_epochs:
            raise RuntimeError(
                "RESUME_EPOCH_INVALID — STOP: "
                f"{epoch}"
            )

        history = state.get("history")
        if not isinstance(history, dict):
            raise RuntimeError(
                "RESUME_HISTORY_INVALID — STOP"
            )

        epochs = history.get("epoch")
        if not isinstance(epochs, list) or not epochs:
            raise RuntimeError(
                "RESUME_HISTORY_INVALID — STOP"
            )

        if int(epochs[-1]) != epoch:
            raise RuntimeError(
                "RESUME_HISTORY_EPOCH_MISMATCH — STOP"
            )

        n = len(epochs)
        for key, value in history.items():
            if isinstance(value, list) and len(value) != n:
                raise RuntimeError(
                    "RESUME_HISTORY_LENGTH_MISMATCH — STOP: "
                    f"{key}"
                )

        self.model.load_state_dict(
            state["model_state_dict"],
            strict=True,
        )
        self.optimizer.load_state_dict(
            state["optimizer_state_dict"]
        )
        self.scheduler.load_state_dict(
            state["scheduler_state_dict"]
        )

        self.patience_counter = int(
            state["patience_counter"]
        )
        self._constant_prediction_streak = int(
            state["constant_prediction_streak"]
        )
        self._protein_param_streak = int(
            state["protein_param_streak"]
        )
        self._drug_param_streak = int(
            state["drug_param_streak"]
        )

        self.best_val_loss = float(
            state["best_val_loss"]
        )
        self.best_epoch = state["best_epoch"]

        self.best_model_state = copy.deepcopy(
            state["best_model_state"]
        )
        self.best_optimizer_state = copy.deepcopy(
            state["best_optimizer_state"]
        )
        self.best_scheduler_state = copy.deepcopy(
            state["best_scheduler_state"]
        )

        self.stop_epoch = state.get("stop_epoch")
        self.stop_reason = str(
            state.get(
                "stop_reason",
                "max_epochs_reached",
            )
        )

        self.history = copy.deepcopy(history)

        # Restore RNG last. Construction/loading above must not
        # perturb the resumed stochastic trajectory.
        self._restore_rng_state(
            state["rng_state"]
        )

        return epoch

    def fit(
        self,
        train_loader,
        val_loader,
        *,
        last_state_path=None,
        protocol_metadata=None,
        resume_key=None,
    ):
        self.stop_reason = "max_epochs_reached"

        last_completed_epoch = 0

        if last_state_path is not None:
            last_state_path = os.fspath(
                last_state_path
            )

            if resume_key is None:
                raise RuntimeError(
                    "RESUME_KEY_MISSING — STOP"
                )

            if os.path.exists(last_state_path):
                last_completed_epoch = (
                    self._load_last_state(
                        last_state_path,
                        protocol_metadata,
                        resume_key,
                    )
                )
                print(
                    "RESUME LAST_STATE: "
                    f"completed_epoch="
                    f"{last_completed_epoch}; "
                    f"next_epoch="
                    f"{last_completed_epoch + 1}"
                )

        start_epoch = last_completed_epoch + 1

        # A LAST_STATE may have been committed immediately
        # before the normal early-stop check or at max_epochs.
        # In either case, do not grant extra epochs after restart.
        if self.patience_counter >= self.patience:
            self.stop_epoch = int(
                last_completed_epoch
            )
            self.stop_reason = (
                "early_stopping_patience"
            )
            start_epoch = self.num_epochs + 1

        elif last_completed_epoch >= self.num_epochs:
            self.stop_epoch = int(
                last_completed_epoch
            )
            self.stop_reason = (
                "max_epochs_reached"
            )
            start_epoch = self.num_epochs + 1

        for epoch in range(
            start_epoch,
            self.num_epochs + 1,
        ):
            lr_used = float(self.optimizer.param_groups[0]["lr"])
            train_loss = self.train_epoch(train_loader)
            val_loss, val_pred, _ = self.evaluate(val_loader)

            val_prediction_std = float(
                np.std(np.asarray(val_pred, dtype=float))
            )
            if not np.isfinite(val_prediction_std):
                raise FloatingPointError(
                    "Non-finite validation prediction std"
                )

            param_snapshot = self._parameter_snapshot()

            self.history["epoch"].append(int(epoch))
            self.history["train_loss"].append(float(train_loss))
            self.history["val_loss"].append(float(val_loss))
            self.history["lr"].append(lr_used)
            self.history["val_prediction_std"].append(
                val_prediction_std
            )
            self.history["total_grad_norm"].append(
                float(
                    self._epoch_grad_snapshot.get(
                        "total_grad_norm", float("nan")
                    )
                )
            )
            self.history["protein_grad_norm"].append(
                float(
                    self._epoch_grad_snapshot.get(
                        "protein_grad_norm", float("nan")
                    )
                )
            )
            self.history["drug_grad_norm"].append(
                float(
                    self._epoch_grad_snapshot.get(
                        "drug_grad_norm", float("nan")
                    )
                )
            )

            for key in (
                "protein_param_norm",
                "drug_param_norm",
                "protein_embedding_norm",
                "protein_token_proj_norm",
                "drug_input_proj_norm",
            ):
                self.history[key].append(
                    float(param_snapshot[key])
                )

            print(
                f"Epoch {epoch:3d}/{self.num_epochs} | "
                f"train MSE={train_loss:.6f} | "
                f"val MSE={val_loss:.6f} | "
                f"pred_std={val_prediction_std:.3e} | "
                f"lr={lr_used:.2e}"
            )

            print(
                "  HEALTH | "
                f"grad(total/protein/drug)="
                f"{self.history['total_grad_norm'][-1]:.3e}/"
                f"{self.history['protein_grad_norm'][-1]:.3e}/"
                f"{self.history['drug_grad_norm'][-1]:.3e} | "
                f"param(protein/drug)="
                f"{param_snapshot['protein_param_norm']:.3e}/"
                f"{param_snapshot['drug_param_norm']:.3e} | "
                f"protein(embed/proj)="
                f"{param_snapshot['protein_embedding_norm']:.3e}/"
                f"{param_snapshot['protein_token_proj_norm']:.3e}"
            )

            # Constant-output hard guard.
            if epoch >= self.constant_prediction_start_epoch:
                if val_prediction_std <= self.constant_prediction_std_eps:
                    self._constant_prediction_streak += 1
                else:
                    self._constant_prediction_streak = 0

                if (
                    self._constant_prediction_streak
                    >= self.constant_prediction_patience
                ):
                    self.stop_epoch = int(epoch)
                    self.stop_reason = "constant_output_collapse"
                    raise RuntimeError(
                        "CONSTANT_OUTPUT_COLLAPSE: validation "
                        f"prediction std <= "
                        f"{self.constant_prediction_std_eps:g} for "
                        f"{self._constant_prediction_streak} "
                        "consecutive epochs"
                    )

            # Canonical branch parameter-collapse guard. Baselines that do
            # not expose these exact modules are intentionally not killed by
            # this branch-specific rule.
            if epoch >= self.branch_param_start_epoch:
                p_embed = param_snapshot[
                    "protein_embedding_norm"
                ]
                p_proj = param_snapshot[
                    "protein_token_proj_norm"
                ]
                d_input = param_snapshot[
                    "drug_input_proj_norm"
                ]

                p_bad = (
                    np.isfinite(p_embed)
                    and np.isfinite(p_proj)
                    and (
                        p_embed <= self.branch_param_norm_eps
                        or p_proj <= self.branch_param_norm_eps
                    )
                )
                d_bad = (
                    np.isfinite(d_input)
                    and d_input <= self.branch_param_norm_eps
                )

                self._protein_param_streak = (
                    self._protein_param_streak + 1
                    if p_bad else 0
                )
                self._drug_param_streak = (
                    self._drug_param_streak + 1
                    if d_bad else 0
                )

                if (
                    self._protein_param_streak
                    >= self.branch_param_patience
                ):
                    self.stop_epoch = int(epoch)
                    self.stop_reason = "protein_parameter_collapse"
                    raise RuntimeError(
                        "PROTEIN_BRANCH_COLLAPSE: critical "
                        "protein parameter norm remained near zero"
                    )

                if (
                    self._drug_param_streak
                    >= self.branch_param_patience
                ):
                    self.stop_epoch = int(epoch)
                    self.stop_reason = "drug_parameter_collapse"
                    raise RuntimeError(
                        "DRUG_BRANCH_COLLAPSE: critical drug "
                        "parameter norm remained near zero"
                    )

            improved = self._is_improvement(val_loss)
            if improved:
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.scheduler.step(val_loss)

            if improved:
                self._capture_best_state(epoch, val_loss)

            # Commit an atomic LAST_STATE only after the epoch has
            # fully completed: validation, health telemetry,
            # patience update, scheduler step and best-state update.
            if last_state_path is not None:
                self._save_last_state(
                    last_state_path,
                    epoch,
                    protocol_metadata,
                    resume_key,
                )

            if self.patience_counter >= self.patience:
                self.stop_epoch = int(epoch)
                self.stop_reason = "early_stopping_patience"
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best epoch={self.best_epoch}, "
                    f"best val MSE={self.best_val_loss:.6f}"
                )
                break

        if not self.history["epoch"]:
            raise RuntimeError("Training produced no epochs")
        if self.stop_epoch is None:
            self.stop_epoch = int(self.history["epoch"][-1])
        if self.best_model_state is None:
            raise RuntimeError("No best validation state was captured")

        self.history["best_epoch"] = self.best_epoch
        self.history["best_val_loss"] = self.best_val_loss
        self.history["stop_epoch"] = self.stop_epoch
        self.history["stop_reason"] = self.stop_reason

        self.model.load_state_dict(self.best_model_state, strict=True)
        if self.best_optimizer_state is not None:
            self.optimizer.load_state_dict(self.best_optimizer_state)
        if self.best_scheduler_state is not None:
            self.scheduler.load_state_dict(self.best_scheduler_state)
        return self.history

    def save(
        self,
        path="checkpoints/best_model.pt",
        *,
        scaler=None,
        history=None,
        fusion_mode=None,
        edge_dim=None,
        dataset=None,
        setting=None,
        model_seed=None,
        split_metadata=None,
        data_provenance=None,
        validation_metrics=None,
        extra_metadata=None,
        protocol_metadata=None,
    ):
        """Atomically save the best-validation checkpoint.

        Final test metrics are intentionally absent. They live in separate
        evaluation artifacts created only during the explicit test phase.
        """
        path = os.fspath(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if history is None:
            history = self.history

        save_dict: Dict[str, Any] = {
            "checkpoint_format_version": 5,
            "architecture_version": getattr(self.model, "architecture_version", None),
            "optimizer": str(self.config.get("TRAINING", {}).get("optimizer", "")).lower(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_state_epoch": self.best_epoch,
            "best_val_loss": float(self.best_val_loss),
            "best_epoch": self.best_epoch,
            "stop_epoch": self.stop_epoch,
            "stop_reason": self.stop_reason,
            "history": history,
            "config": copy.deepcopy(self.config),
            "model_seed": int(model_seed if model_seed is not None else self.model_seed),
            "parameter_summary": (
                self.model.parameter_summary()
                if hasattr(self.model, "parameter_summary")
                else {
                    "total": sum(
                        p.numel() for p in self.model.parameters() if p.requires_grad
                    )
                }
            ),
            "test_evaluated_during_training": False,
        }

        optional = {
            "scaler": scaler,
            "fusion_mode": fusion_mode,
            "edge_dim": edge_dim,
            "dataset": dataset,
            "setting": setting,
            "split_metadata": split_metadata,
            "data_provenance": data_provenance,
            "validation_metrics": validation_metrics,
            "extra_metadata": extra_metadata,
        }
        for key, value in optional.items():
            if value is not None:
                save_dict[key] = value

        if protocol_metadata is not None:
            required_protocol_keys = (
                "protocol_version",
                "optimizer",
                "config_sha256",
                "source_sha256",
                "data_manifest_sha256",
                "training_data_sha256",
                "plan_sha256",
            )
            missing = [k for k in required_protocol_keys if not protocol_metadata.get(k)]
            if missing:
                raise ValueError(
                    "Incomplete protocol_metadata for formal checkpoint: "
                    + ", ".join(missing)
                )
            if str(protocol_metadata["optimizer"]).lower() != save_dict["optimizer"]:
                raise ValueError(
                    "protocol optimizer does not match Trainer config optimizer"
                )
            for key in required_protocol_keys:
                save_dict[key] = protocol_metadata[key]
            if protocol_metadata.get("plan_path") is not None:
                save_dict["plan_path"] = protocol_metadata["plan_path"]
            if protocol_metadata.get("plan_family") is not None:
                save_dict["plan_family"] = protocol_metadata["plan_family"]

        tmp_path = path + ".tmp"
        torch.save(save_dict, tmp_path)
        os.replace(tmp_path, path)

        print(f"Saved best-validation checkpoint: {path}")
        print(f"  architecture={save_dict.get('architecture_version')}")
        print(
            f"  best_epoch={self.best_epoch}, stop_epoch={self.stop_epoch}, "
            f"stop_reason={self.stop_reason}"
        )
        print("  test evaluated during training: NO")
        return path
