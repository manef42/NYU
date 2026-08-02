"""Unified regressor interface and registry for every FSR model family.

Five families are compared: XGBoost, Random Forest, k-nearest neighbours, a
multi-layer perceptron, and Support Vector Regression. Each family uses
whatever accelerator is available - XGBoost trains with the CUDA histogram
builder, the MLP trains in PyTorch on one GPU or across every visible GPU with
NCCL DistributedDataParallel, and KNN uses cuML when it is installed - while
remaining bit-for-bit runnable on CPU-only hosts so that results are
reproducible off the cluster.
"""
from __future__ import annotations

import os
import socket
import tempfile
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from xgboost import XGBRegressor

from fsr_common import (RANDOM_STATE, build_preprocessor, configure_torch,
                        empty_cuda_cache, feature_lists, gpu_count,
                        inverse_signed_log1p, paper_weights, signed_log1p,
                        slurm_cpu_count, slurm_num_workers, torch_device,
                        xgboost_device)

XGB_OBJECTIVES = {
    "squarederror": "reg:squarederror",
    "absoluteerror": "reg:absoluteerror",
    "pseudohuber": "reg:pseudohubererror",
}


# =============================================================================
# PyTorch multi-layer perceptron with GPU and multi-GPU training
# =============================================================================

class _TorchNetwork(torch.nn.Module):
    def __init__(self, input_size: int, hidden_sizes: tuple[int, ...], activation: str):
        super().__init__()
        activation_factory: type[torch.nn.Module] = (
            torch.nn.ReLU if activation == "relu" else torch.nn.Tanh)
        layers: list[torch.nn.Module] = []
        width = input_size
        for hidden in hidden_sizes:
            layers.extend((torch.nn.Linear(width, hidden), activation_factory()))
            width = hidden
        layers.append(torch.nn.Linear(width, 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(1)


def _train_network(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                   learning_rate: float, weight_decay: float, max_iter: int,
                   patience: int = 25, sampler: DistributedSampler | None = None) -> None:
    """Weighted mean-squared-error training with deterministic early stopping."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_loss, stale_epochs = float("inf"), 0
    for epoch in range(max_iter):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        totals = torch.zeros(2, device=device)
        for features, targets, weights in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            squared = (model(features) - targets) ** 2
            loss = (weights * squared).sum() / weights.sum().clamp_min(1e-12)
            loss.backward()
            optimizer.step()
            totals[0] += loss.detach() * weights.sum()
            totals[1] += weights.sum()
        if dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        epoch_loss = float((totals[0] / totals[1].clamp_min(1e-12)).item())
        if epoch_loss < best_loss - 1e-8:
            best_loss, stale_epochs = epoch_loss, 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break


def _ddp_worker(rank: int, world_size: int, port: int, features: np.ndarray,
                targets: np.ndarray, weights: np.ndarray, settings: dict[str, Any],
                state_path: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        device = configure_torch(rank)
        dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets),
                                torch.from_numpy(weights))
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=RANDOM_STATE)
        loader = DataLoader(dataset, batch_size=settings["batch_size"], sampler=sampler,
                           pin_memory=True, num_workers=slurm_num_workers())
        network = _TorchNetwork(features.shape[1], settings["hidden_layer_sizes"],
                               settings["activation"]).to(device)
        model = DistributedDataParallel(network, device_ids=[rank], output_device=rank)
        _train_network(model, loader, device, settings["learning_rate_init"],
                       settings["alpha"], settings["max_iter"], sampler=sampler)
        if rank == 0:
            torch.save({key: value.detach().cpu()
                        for key, value in model.module.state_dict().items()}, state_path)
        dist.barrier()
    finally:
        dist.destroy_process_group()
        empty_cuda_cache()


class TorchMLPRegressor:
    """Small scikit-learn-style PyTorch MLP with GPU and conditional DDP training."""

    def __init__(self, hidden_layer_sizes: tuple[int, ...] = (100,), alpha: float = 1e-4,
                 learning_rate_init: float = 1e-3, activation: str = "relu",
                 batch_size: int = 32, max_iter: int = 600,
                 random_state: int = RANDOM_STATE, **_: Any):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.alpha = float(alpha)
        self.learning_rate_init = float(learning_rate_init)
        self.activation = activation
        self.batch_size = int(batch_size)
        self.max_iter = int(max_iter)
        self.random_state = RANDOM_STATE
        self.input_size_: int | None = None
        self.state_dict_: dict[str, torch.Tensor] | None = None

    def _settings(self) -> dict[str, Any]:
        return {"hidden_layer_sizes": self.hidden_layer_sizes, "alpha": self.alpha,
                "learning_rate_init": self.learning_rate_init,
                "activation": self.activation, "batch_size": self.batch_size,
                "max_iter": self.max_iter}

    @staticmethod
    def _open_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> "TorchMLPRegressor":
        torch.manual_seed(RANDOM_STATE)
        features = np.ascontiguousarray(X, dtype=np.float32)
        targets = np.ascontiguousarray(y, dtype=np.float32)
        weights = np.ascontiguousarray(
            np.ones(len(targets)) if sample_weight is None else sample_weight,
            dtype=np.float32)
        self.input_size_ = int(features.shape[1])
        if gpu_count() > 1:
            with tempfile.TemporaryDirectory(prefix="fsr-ddp-") as directory:
                state_path = os.path.join(directory, "state.pt")
                mp.spawn(_ddp_worker,
                         args=(gpu_count(), self._open_port(), features, targets, weights,
                               self._settings(), state_path),
                         nprocs=gpu_count(), join=True)
                self.state_dict_ = torch.load(state_path, map_location="cpu",
                                              weights_only=True)
            return self
        device = configure_torch()
        dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets),
                                torch.from_numpy(weights))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                           pin_memory=torch.cuda.is_available(),
                           num_workers=slurm_num_workers() if torch.cuda.is_available() else 0)
        model = _TorchNetwork(
            self.input_size_, self.hidden_layer_sizes, self.activation).to(device)
        _train_network(model, loader, device, self.learning_rate_init, self.alpha,
                       self.max_iter)
        self.state_dict_ = {key: value.detach().cpu()
                           for key, value in model.state_dict().items()}
        empty_cuda_cache()
        return self

    def _model(self, device: torch.device) -> _TorchNetwork:
        if self.input_size_ is None or self.state_dict_ is None:
            raise RuntimeError("MLP is not fitted")
        model = _TorchNetwork(self.input_size_, self.hidden_layer_sizes, self.activation)
        model.load_state_dict(self.state_dict_)
        return model.to(device).eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        device = torch_device()
        model = self._model(device)
        loader = DataLoader(
            TensorDataset(torch.as_tensor(np.asarray(X, dtype=np.float32))),
            batch_size=max(self.batch_size, 256), shuffle=False,
            pin_memory=torch.cuda.is_available(), num_workers=0)
        outputs = []
        with torch.no_grad():
            for (batch,) in loader:
                outputs.append(model(batch.to(device, non_blocking=True)).cpu().numpy())
        empty_cuda_cache()
        return np.concatenate(outputs)


class CuMLKNNWithFallback:
    """Prefer a GPU cuML nearest-neighbour regressor, else threaded scikit-learn."""

    def __init__(self, params: dict[str, Any]):
        self.params = dict(params)
        self.estimator_: Any = None
        self.using_cuml_ = False

    def _cuml_compatible(self) -> bool:
        return (gpu_count() > 0 and self.params.get("weights", "uniform") == "uniform"
                and int(self.params.get("p", 2)) == 2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CuMLKNNWithFallback":
        if self._cuml_compatible():
            try:
                from cuml.neighbors import KNeighborsRegressor as CuMLKNN
                estimator = CuMLKNN(n_neighbors=int(self.params["n_neighbors"]))
                estimator.fit(np.asarray(X, dtype=np.float32),
                              np.asarray(y, dtype=np.float32))
                self.estimator_, self.using_cuml_ = estimator, True
                return self
            except (ImportError, ModuleNotFoundError, TypeError, ValueError,
                    RuntimeError) as exc:
                warnings.warn(f"cuML KNN unavailable; using scikit-learn KNN: {exc}",
                              RuntimeWarning)
        self.estimator_ = KNeighborsRegressor(n_jobs=slurm_cpu_count(), **self.params)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.estimator_ is None:
            raise RuntimeError("KNN is not fitted")
        prediction = self.estimator_.predict(X)
        if hasattr(prediction, "to_numpy"):
            prediction = prediction.to_numpy()
        elif hasattr(prediction, "get"):
            prediction = prediction.get()
        return np.asarray(prediction, dtype=float).ravel()


# =============================================================================
# Registry
# =============================================================================

@dataclass(frozen=True)
class ModelSpec:
    family: str
    estimator_factory: Callable[[dict[str, Any]], Any]
    search_space: dict[str, list[Any]]
    supports_sample_weight: bool
    scale_numeric: bool
    native_numeric_missing: bool
    accelerator: str
    deterministic_note: str


def _xgb(params: dict[str, Any]) -> XGBRegressor:
    params = dict(params)
    objective = str(params.pop("objective_variant", "squarederror"))
    if objective not in XGB_OBJECTIVES:
        raise ValueError(f"Unknown XGBoost objective variant {objective!r}")
    return XGBRegressor(objective=XGB_OBJECTIVES[objective], tree_method="hist",
                        device=xgboost_device(), n_jobs=slurm_cpu_count(),
                        random_state=RANDOM_STATE, verbosity=0, **params)


def _rf(params: dict[str, Any]) -> RandomForestRegressor:
    return RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=slurm_cpu_count(),
        **params,
    )


def _knn(params: dict[str, Any]) -> CuMLKNNWithFallback:
    return CuMLKNNWithFallback(params)


def _mlp(params: dict[str, Any]) -> TorchMLPRegressor:
    return TorchMLPRegressor(random_state=RANDOM_STATE, **params)


def _svr(params: dict[str, Any]) -> SVR:
    return SVR(**params)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "xgboost": ModelSpec(
        "xgboost", _xgb, {
            "n_estimators": [100, 200, 300, 500, 800],
            "learning_rate": [.01, .02, .05, .1, .2],
            "max_depth": [2, 3, 4, 5, 6],
            "min_child_weight": [1, 2, 4, 6, 10],
            "subsample": [.6, .7, .8, .9, 1.],
            "colsample_bytree": [.5, .7, .8, 1.],
            "reg_lambda": [.5, 1., 2., 5.],
            "reg_alpha": [0., .1, .5, 1.],
            "gamma": [0., .1, .3, 1.],
        }, True, False, True, "gpu",
        "Random state 42; CUDA histogram tree method when a GPU is visible."),
    "random_forest": ModelSpec(
        "random_forest", _rf, {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [4, 6, 8, 10, 12, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8, 16],
            "max_features": ["sqrt", "log2", .5, .8, 1.0],
        }, True, False, False, "cpu",
        "Random state 42; parallelized over all allocated CPU cores."),
    "knn": ModelSpec(
        "knn", _knn, {
            "n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        }, False, True, False, "gpu-optional",
        "Deterministic; cuML GPU neighbours when compatible, else all CPU cores."),
    "mlp": ModelSpec(
        "mlp", _mlp, {
            "hidden_layer_sizes": [(100,), (128, 64), (100, 50), (64, 64, 32), (256, 128)],
            "activation": ["relu", "tanh"],
            "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
            "learning_rate_init": [1e-3, 5e-3, 1e-2],
            "batch_size": [32, 64, 128],
        }, True, True, False, "gpu",
        "PyTorch seed 42; one GPU or NCCL DDP across every visible GPU."),
    "svr": ModelSpec(
        "svr", _svr, {
            "C": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
            "epsilon": [0.01, 0.05, 0.1, 0.2, 0.5],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        }, False, True, False, "cpu",
        "sklearn SVR; no sample_weight support, uses weighted resampling instead."),
}

# Ordered tuple used by fsr_evaluate.py --model-id Slurm array indexing.
# Index: 0=xgboost, 1=random_forest, 2=knn, 3=mlp, 4=svr
MODEL_ID_TO_FAMILY = ("xgboost", "random_forest", "knn", "mlp", "svr")


# =============================================================================
# Model wrapper
# =============================================================================

class FSRModel:
    """Preprocessing, weighting, optional target compression, and one estimator."""

    def __init__(self, family: str, feature_set: str, params: dict[str, Any],
                 manifest: list[dict[str, str]], paper_weight: str = "none",
                 target_transform: str = "identity", monotone_oxygen: bool = False,
                 paper_bagging: int = 0):
        if family not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model family {family!r}")
        if target_transform not in {"identity", "signed_log1p"}:
            raise ValueError(f"Unknown target transform {target_transform!r}")
        self.family, self.spec = family, MODEL_REGISTRY[family]
        self.feature_set = feature_set
        self.params = dict(params)
        self.manifest = manifest
        self.paper_weight = paper_weight
        self.target_transform = target_transform
        self.monotone_oxygen = bool(monotone_oxygen)
        self.paper_bagging = int(paper_bagging)
        if self.monotone_oxygen and family != "xgboost":
            raise ValueError("Monotone oxygen constraints are only defined for XGBoost")
        if self.paper_bagging and family != "xgboost":
            raise ValueError("Paper-cluster bagging is restricted to XGBoost candidates")
        self.numeric, self.categorical = feature_lists(manifest, feature_set)
        self.features = self.numeric + self.categorical
        self.preprocessor_: ColumnTransformer | None = None
        self.estimators_: list[Any] = []

    def _preprocessor(self) -> ColumnTransformer:
        return build_preprocessor(
            self.numeric, self.categorical, scale_numeric=self.spec.scale_numeric,
            native_numeric_missing=self.spec.native_numeric_missing)

    def _forward_target(self, y: np.ndarray) -> np.ndarray:
        return signed_log1p(y) if self.target_transform == "signed_log1p" else np.asarray(
            y, dtype=float)

    def _inverse_target(self, y: np.ndarray) -> np.ndarray:
        return (inverse_signed_log1p(y) if self.target_transform == "signed_log1p"
                else np.asarray(y, dtype=float))

    def _monotone_constraints(self, width: int) -> str:
        oxygen = [index for index, name in enumerate(self.numeric)
                  if "oxygen" in str(name).lower()]
        if not oxygen:
            raise ValueError("No oxygen feature is available for a monotone constraint")
        constraints = [0] * width
        for index in oxygen:
            constraints[index] = 1
        return "(" + ",".join(map(str, constraints)) + ")"

    def _make_estimator(self, width: int) -> Any:
        params = dict(self.params)
        if self.monotone_oxygen:
            params["monotone_constraints"] = self._monotone_constraints(width)
        return self.spec.estimator_factory(params)

    def _resample(self, weights: np.ndarray) -> np.ndarray:
        """Deterministic probability-proportional resampling for weight-blind models."""
        rng = np.random.default_rng(RANDOM_STATE)
        probabilities = weights / weights.sum()
        return rng.choice(np.arange(len(weights)), size=len(weights), replace=True,
                          p=probabilities)

    def fit(self, X: pd.DataFrame, y: np.ndarray, papers: pd.Series) -> "FSRModel":
        missing = [column for column in self.features if column not in X.columns]
        if missing:
            raise ValueError(f"Training frame is missing feature columns: {missing}")
        y = np.asarray(y, dtype=float)
        papers = pd.Series(papers).reset_index(drop=True)
        self.preprocessor_ = self._preprocessor()
        transformed = np.asarray(
            self.preprocessor_.fit_transform(X[self.features]), dtype=np.float32)
        target = self._forward_target(y)
        weights = paper_weights(papers, self.paper_weight)
        self.estimators_ = []
        if self.paper_bagging:
            rng = np.random.default_rng(RANDOM_STATE)
            unique = papers.unique()
            rows = {paper: np.flatnonzero(papers.to_numpy() == paper) for paper in unique}
            partitions = [np.concatenate([rows[paper] for paper in
                                          rng.choice(unique, len(unique), replace=True)])
                          for _ in range(self.paper_bagging)]
        else:
            partitions = [np.arange(len(target))]
        for indices in partitions:
            estimator = self._make_estimator(transformed.shape[1])
            if self.spec.supports_sample_weight:
                estimator.fit(transformed[indices], target[indices],
                              sample_weight=weights[indices])
            else:
                sampled = indices[self._resample(weights[indices])]
                estimator.fit(transformed[sampled], target[sampled])
            self.estimators_.append(estimator)
            empty_cuda_cache()
        if not self.estimators_:
            raise RuntimeError("No estimator was fitted")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.preprocessor_ is None or not self.estimators_:
            raise RuntimeError("Model is not fitted")
        transformed = np.asarray(
            self.preprocessor_.transform(X[self.features]), dtype=np.float32)
        predictions = [np.asarray(estimator.predict(transformed), dtype=float).ravel()
                       for estimator in self.estimators_]
        result = self._inverse_target(np.mean(predictions, axis=0))
        empty_cuda_cache()
        if not np.all(np.isfinite(result)):
            raise ValueError("Estimator produced non-finite flame-spread-rate predictions")
        return result

    def feature_importances(self) -> np.ndarray | None:
        """Mean impurity importance over the fitted estimators, when defined."""
        if not self.estimators_ or not all(
                hasattr(estimator, "feature_importances_") for estimator in self.estimators_):
            return None
        return np.mean([np.asarray(estimator.feature_importances_, dtype=float)
                        for estimator in self.estimators_], axis=0)

    def get_feature_names_out(self) -> np.ndarray:
        if self.preprocessor_ is None:
            raise RuntimeError("Model is not fitted")
        names = self.preprocessor_.get_feature_names_out()
        return np.array([str(name).replace("numeric__", "").replace("categorical__", "")
                         for name in names])


def make_model(candidate: dict[str, Any], params: dict[str, Any] | None,
               manifest: list[dict[str, str]]) -> FSRModel:
    merged = dict(candidate.get("fixed_params", {}))
    merged.update(params or {})
    return FSRModel(
        family=candidate["model_family"], feature_set=candidate.get("feature_set", "all"),
        params=merged, manifest=manifest,
        paper_weight=candidate.get("paper_weight", "none"),
        target_transform=candidate.get("target_transform", "identity"),
        monotone_oxygen=bool(candidate.get("monotone_oxygen", False)),
        paper_bagging=int(candidate.get("paper_bagging", 0) or 0),
    )
