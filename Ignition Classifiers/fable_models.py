"""Unified model interface and registry for all classifier families."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from fable_common import combined_weights, feature_lists, torch_device


RANDOM_STATE = 42


@dataclass(frozen=True)
class FocalObjective:
    """Pickle-safe stable binary focal-loss objective for XGBoost."""
    gamma: float

    def __call__(
        self,
        y_true: np.ndarray,
        raw: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        y = y_true.astype(float)
        p = np.clip(1 / (1 + np.exp(-np.clip(raw, -30, 30))), 1e-7, 1 - 1e-7)
        pt = np.where(y == 1, p, 1 - p)
        sign = np.where(y == 1, -1.0, 1.0)
        grad = sign * (1 - pt) ** self.gamma * (
            self.gamma * pt * np.log(pt) - (1 - pt)
        )
        hess = (1 - pt) ** self.gamma * pt * (
            (1 - pt) + self.gamma * (
                -np.log(pt) * (1 - (self.gamma + 1) * pt) + 2 * (1 - pt)
            )
        )
        hess = np.maximum(hess, 1e-6)
        if sample_weight is not None:
            grad, hess = grad * sample_weight, hess * sample_weight
        return grad, hess


def focal_objective(gamma: float) -> FocalObjective:
    return FocalObjective(gamma)


class _TorchNetwork(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...],
        activation: str,
    ):
        super().__init__()
        activation_factory: type[torch.nn.Module] = (
            torch.nn.ReLU if activation == "relu" else torch.nn.Tanh
        )
        layers: list[torch.nn.Module] = []
        width = input_size

        for hidden in hidden_sizes:
            layers.extend((torch.nn.Linear(width, hidden), activation_factory()))
            width = hidden

        layers.append(torch.nn.Linear(width, 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(1)


def _train_network(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    max_iter: int,
) -> None:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    best_loss, stale_epochs = float("inf"), 0

    for _epoch in range(max_iter):
        model.train()
        total_loss = torch.zeros(2, device=device)

        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()

            total_loss[0] += loss.detach() * len(targets)
            total_loss[1] += len(targets)

        epoch_loss = float(
            (total_loss[0] / total_loss[1].clamp_min(1)).item()
        )

        if epoch_loss < best_loss - 1e-6:
            best_loss, stale_epochs = epoch_loss, 0
        else:
            stale_epochs += 1

        if stale_epochs >= 20:
            break


class TorchMLPClassifier:
    """Small sklearn-compatible PyTorch MLP."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (100,),
        alpha: float = 0.0001,
        learning_rate_init: float = 0.001,
        activation: str = "relu",
        batch_size: int = 32,
        max_iter: int = 600,
        random_state: int = RANDOM_STATE,
        **_: Any,
    ):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.activation = activation
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.input_size_: int | None = None
        self.state_dict_: dict[str, torch.Tensor] | None = None
        self.classes_ = np.array([0, 1])

    def _settings(self) -> dict[str, Any]:
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "alpha": self.alpha,
            "learning_rate_init": self.learning_rate_init,
            "activation": self.activation,
            "batch_size": self.batch_size,
            "max_iter": self.max_iter,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLPClassifier":
        torch.manual_seed(self.random_state)
        features = np.ascontiguousarray(X, dtype=np.float32)
        targets = np.ascontiguousarray(y, dtype=np.float32)
        self.input_size_ = features.shape[1]

        device = torch_device()
        dataset = TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(targets),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        model = _TorchNetwork(
            self.input_size_,
            self.hidden_layer_sizes,
            self.activation,
        ).to(device)

        _train_network(
            model,
            loader,
            device,
            self.learning_rate_init,
            self.alpha,
            self.max_iter,
        )

        self.state_dict_ = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }
        return self

    def _model(self, device: torch.device) -> _TorchNetwork:
        if self.input_size_ is None or self.state_dict_ is None:
            raise RuntimeError("MLP is not fitted")

        model = _TorchNetwork(
            self.input_size_,
            self.hidden_layer_sizes,
            self.activation,
        )
        model.load_state_dict(self.state_dict_)
        return model.to(device).eval()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        device = torch_device()
        features = torch.as_tensor(np.asarray(X, dtype=np.float32))
        loader = DataLoader(
            TensorDataset(features),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        model = self._model(device)
        probabilities: list[np.ndarray] = []

        with torch.no_grad():
            for (batch,) in loader:
                logits = model(batch.to(device, non_blocking=True))
                probabilities.append(torch.sigmoid(logits).cpu().numpy())

        positive = np.concatenate(probabilities)
        return np.column_stack((1 - positive, positive))


class CuMLSVCWithFallback:
    """Calibrated sklearn SVM; use LinearSVC for linear-kernel candidates."""

    def __init__(self, params: dict[str, Any], random_state: int = RANDOM_STATE):
        self.params = dict(params)
        self.random_state = random_state
        self.estimator_: Any | None = None
        self.using_cuml_ = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "CuMLSVCWithFallback":
        params = dict(self.params)
        kernel = params.pop("kernel", "rbf")

        if kernel == "linear":
            # Generic SVC-grid parameters that are irrelevant to LinearSVC.
            params.pop("gamma", None)
            params.pop("shrinking", None)
            params.pop("cache_size", None)
            params.pop("probability", None)

            base_estimator = LinearSVC(
                C=float(params.pop("C", 1.0)),
                dual="auto",
                max_iter=10_000,
                tol=1e-3,
                random_state=self.random_state,
                **params,
            )
        else:
            base_estimator = SVC(
                kernel=kernel,
                cache_size=4000,
                tol=1e-3,
                max_iter=10_000,
                random_state=self.random_state,
                **params,
            )

        estimator = CalibratedClassifierCV(
            base_estimator,
            method="sigmoid",
            cv=3,
            n_jobs=-1,
        )
        estimator.fit(X, y, sample_weight=sample_weight)
        self.estimator_ = estimator
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.estimator_ is None:
            raise RuntimeError("SVM is not fitted")

        probability = self.estimator_.predict_proba(X)
        if hasattr(probability, "to_numpy"):
            probability = probability.to_numpy()
        elif hasattr(probability, "get"):
            probability = probability.get()
        return np.asarray(probability)


@dataclass(frozen=True)
class ModelSpec:
    family: str
    estimator_factory: Callable[[dict[str, Any], int], Any]
    search_space: dict[str, list[Any]]
    supports_sample_weight: bool
    scale_numeric: bool
    native_numeric_missing: bool
    deterministic_note: str


def _xgb(params: dict[str, Any], seed: int) -> XGBClassifier:
    objective = params.pop("objective_variant", "logistic")
    focal_gamma = float(params.pop("focal_gamma", 2.0))
    kwargs = {
        "tree_method": "hist",
        "device": "cpu",
        "eval_metric": "logloss",
        "n_jobs": -1,
        "random_state": seed,
        **params,
    }

    if objective == "focal":
        kwargs.update(objective=focal_objective(focal_gamma), base_score=0.5)
    else:
        kwargs["objective"] = "binary:logistic"

    return XGBClassifier(**kwargs)


def _knn(params: dict[str, Any], _: int) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_jobs=-1, **params)


def _tree(params: dict[str, Any], seed: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=seed, **params)


def _mlp(params: dict[str, Any], seed: int) -> TorchMLPClassifier:
    return TorchMLPClassifier(max_iter=400, random_state=seed, **params)


def _svm(params: dict[str, Any], seed: int) -> CuMLSVCWithFallback:
    return CuMLSVCWithFallback(params, random_state=seed)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "xgboost": ModelSpec(
        "xgboost",
        _xgb,
        {
            "n_estimators": [200, 400, 600, 800],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "max_depth": [2, 3, 4, 5, 6],
            "min_child_weight": [1, 2, 4, 8],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
            "reg_alpha": [0.0, 0.1, 0.5],
            "gamma": [0.0, 0.1, 0.5],
        },
        True,
        False,
        True,
        "Random state fixed; CPU histogram tree method.",
    ),
    "knn": ModelSpec(
        "knn",
        _knn,
        {
            "n_neighbors": [5, 9, 15, 25, 40, 60],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
            "leaf_size": [15, 30, 60],
        },
        False,
        True,
        False,
        "Deterministic; all available CPU cores are used.",
    ),
    "decision_tree": ModelSpec(
        "decision_tree",
        _tree,
        {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_leaf": [1, 2, 5, 10, 20],
            "criterion": ["gini", "entropy", "log_loss"],
        },
        True,
        False,
        False,
        "Random state fixed for tie resolution.",
    ),
    "mlp": ModelSpec(
        "mlp",
        _mlp,
        {
            "hidden_layer_sizes": [(32,), (64,), (128, 64), (128, 64, 32)],
            "alpha": [1e-5, 1e-3, 1e-1],
            "learning_rate_init": [1e-3, 3e-3],
            "activation": ["relu"],
            "batch_size": [32, 128],
        },
        False,
        True,
        False,
        "PyTorch seed fixed; CPU or CUDA device selected automatically.",
    ),
    "svm": ModelSpec(
        "svm",
        _svm,
        {
            "C": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
            "gamma": ["scale", 0.01, 0.03, 0.1, 0.3],
            "shrinking": [True, False],
        },
        True,
        True,
        False,
        "Linear candidates use LinearSVC; nonlinear candidates use bounded SVC.",
    ),
}


class FableModel:
    """Model wrapper exposing fit(X, y, papers), predict_proba, and feature names."""

    def __init__(
        self,
        family: str,
        params: dict[str, Any],
        paper_weight: str = "none",
        class_weight: bool = False,
        random_state: int = RANDOM_STATE,
        monotone_oxygen: bool = False,
        paper_bagging: int = 0,
    ):
        if family not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model family {family!r}")

        self.family = family
        self.spec = MODEL_REGISTRY[family]
        self.params = dict(params)
        self.paper_weight = paper_weight
        self.class_weight = class_weight
        self.random_state = random_state
        self.monotone_oxygen = monotone_oxygen
        self.paper_bagging = int(paper_bagging)

        if self.paper_bagging and family != "xgboost":
            raise ValueError(
                "Paper bagging is restricted to scientifically justified "
                "XGBoost candidates"
            )

        self.numeric, self.categorical = feature_lists()
        self.features = self.numeric + self.categorical
        self.preprocessor_: ColumnTransformer | None = None
        self.estimators_: list[Any] = []

    def _preprocessor(self) -> ColumnTransformer:
        numeric_steps: list[tuple[str, Any]] = []

        if not self.spec.native_numeric_missing:
            numeric_steps.append(("imputer", SimpleImputer(strategy="median")))

        if self.spec.scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))

        numeric_transformer: Any = (
            Pipeline(numeric_steps) if numeric_steps else "passthrough"
        )

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        return ColumnTransformer(
            [
                ("numeric", numeric_transformer, self.numeric),
                ("categorical", categorical_transformer, self.categorical),
            ],
            remainder="drop",
            n_jobs=-1,
        )

    def _weights(self, y: np.ndarray, papers: pd.Series) -> np.ndarray:
        return combined_weights(y, papers, self.paper_weight, self.class_weight)

    def _resample(
        self,
        y: np.ndarray,
        weights: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        """Deterministic probability-proportional resampling."""
        rng = np.random.default_rng(seed)
        probabilities = weights / weights.sum()
        indices = rng.choice(
            np.arange(len(y)),
            size=len(y),
            replace=True,
            p=probabilities,
        )

        if len(np.unique(y[indices])) < 2:
            raise ValueError(
                "Weighted resampling produced one class; training fold is invalid"
            )

        return indices

    def _make_estimator(self, y: np.ndarray, width: int, seed: int) -> Any:
        params = dict(self.params)

        if self.family == "xgboost" and self.monotone_oxygen:
            constraints = [0] * width
            constraints[self.numeric.index("oxygen_fraction")] = 1
            params["monotone_constraints"] = (
                "(" + ",".join(map(str, constraints)) + ")"
            )

        return self.spec.estimator_factory(params, seed)

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        papers: pd.Series,
    ) -> "FableModel":
        y = np.asarray(y, dtype=int)
        papers = pd.Series(papers).reset_index(drop=True)

        self.preprocessor_ = self._preprocessor()
        transformed = np.asarray(
            self.preprocessor_.fit_transform(X[self.features]),
            dtype=np.float32,
        )
        weights = self._weights(y, papers)
        self.estimators_ = []

        if self.paper_bagging:
            rng = np.random.default_rng(self.random_state)
            unique = papers.unique()
            rows = {
                paper: np.flatnonzero(papers.to_numpy() == paper)
                for paper in unique
            }
            training_indices = [
                np.concatenate([
                    rows[paper]
                    for paper in rng.choice(unique, len(unique), replace=True)
                ])
                for _ in range(self.paper_bagging)
            ]
        else:
            training_indices = [np.arange(len(y))]

        for indices in training_indices:
            estimator = self._make_estimator(
                y[indices],
                transformed.shape[1],
                self.random_state,
            )

            if self.spec.supports_sample_weight:
                estimator.fit(
                    transformed[indices],
                    y[indices],
                    sample_weight=weights[indices],
                )
            else:
                sampled = self._resample(
                    y[indices],
                    weights[indices],
                    self.random_state,
                )
                estimator.fit(
                    transformed[indices][sampled],
                    y[indices][sampled],
                )

            self.estimators_.append(estimator)

        if not self.estimators_:
            raise RuntimeError("No estimator was fitted")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.preprocessor_ is None or not self.estimators_:
            raise RuntimeError("Model is not fitted")

        transformed = np.asarray(
            self.preprocessor_.transform(X[self.features]),
            dtype=np.float32,
        )
        predictions: list[np.ndarray] = []

        for estimator in self.estimators_:
            if (
                self.family == "xgboost"
                and self.params.get("objective_variant") == "focal"
            ):
                margin = estimator.predict(transformed, output_margin=True)
                probability = 1 / (
                    1 + np.exp(-np.clip(margin, -30, 30))
                )
            else:
                probability = estimator.predict_proba(transformed)[:, 1]

            predictions.append(probability)

        result = np.mean(predictions, axis=0)

        if (
            not np.all(np.isfinite(result))
            or np.any((result < 0) | (result > 1))
        ):
            raise ValueError("Estimator produced invalid class-1 probabilities")

        return result

    def get_feature_names_out(self) -> np.ndarray:
        if self.preprocessor_ is None:
            raise RuntimeError("Model is not fitted")

        return self.preprocessor_.get_feature_names_out()


def make_model(
    candidate: dict[str, Any],
    params: dict[str, Any] | None = None,
    seed: int = RANDOM_STATE,
) -> FableModel:
    merged = dict(candidate.get("fixed_params", {}))
    merged.update(params or {})

    return FableModel(
        family=candidate["model_family"],
        params=merged,
        paper_weight=candidate.get("paper_weight", "none"),
        class_weight=bool(candidate.get("class_weight", False)),
        random_state=seed,
        monotone_oxygen=bool(candidate.get("monotone_oxygen", False)),
        paper_bagging=int(candidate.get("paper_bagging", 0)),
    )