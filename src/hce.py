"""Hidden Consistent Evaluation (HCE) protocol — from AIRA_2 paper.

Splits the agent's training data before it sees any of it:
  - 80% → replaces train.csv (what the agent trains on)
  - 20% → val_features.csv placed in data dir (features only, no labels)

After each run the evaluator scores val_submission.csv against the hidden labels
using the competition metric.  This gives a noise-free signal that doesn't depend
on what CV_SCORE the agent chose to print.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


class HCEEvaluator:
    def __init__(self, val_ratio: float = 0.2, seed: int = 42):
        self.val_ratio = val_ratio
        self.seed = seed
        self._setup_done: bool = False
        self._val_labels: list | None = None
        self._val_ids: list | None = None
        self._id_col: str | None = None
        self._target_col: str | None = None
        self._task_type: str | None = None  # "classification" or "regression"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self, workdir: Path) -> bool:
        """Split train.csv and write val_features.csv.  Returns True on success."""
        try:
            import pandas as pd
        except ImportError:
            logger.warning("HCE: pandas not available, skipping")
            return False

        data_dir = workdir / "home" / "data"
        train_path = data_dir / "train.csv"
        if not train_path.exists():
            logger.info("HCE: train.csv not found — skipping")
            return False

        sample_paths = sorted(data_dir.glob("sample_submission*.csv"))
        if not sample_paths:
            logger.info("HCE: no sample_submission.csv — skipping")
            return False

        try:
            df_train = pd.read_csv(train_path)
            df_sample = pd.read_csv(sample_paths[0])

            if len(df_train) < 20:
                logger.info("HCE: training data too small (%d rows) — skipping", len(df_train))
                return False

            id_col, target_col = self._identify_columns(df_sample, df_train)
            if target_col is None:
                logger.info("HCE: could not identify target column — skipping")
                return False

            # Deterministic split
            rng = random.Random(self.seed)
            indices = list(range(len(df_train)))
            rng.shuffle(indices)
            val_size = max(1, int(len(df_train) * self.val_ratio))
            val_idx = set(indices[:val_size])
            train_idx = [i for i in range(len(df_train)) if i not in val_idx]

            df_val = df_train.iloc[list(val_idx)].reset_index(drop=True)
            df_train_sub = df_train.iloc[train_idx].reset_index(drop=True)

            # Overwrite train.csv with the agent's subset
            df_train_sub.to_csv(train_path, index=False)

            # Write val_features.csv (no target — agent must not see labels)
            feature_cols = [c for c in df_val.columns if c != target_col]
            df_val[feature_cols].to_csv(data_dir / "val_features.csv", index=False)

            # Store hidden labels
            self._val_labels = df_val[target_col].tolist()
            self._val_ids = df_val[id_col].tolist() if id_col else list(range(len(df_val)))
            self._id_col = id_col
            self._target_col = target_col
            self._task_type = self._detect_task_type(df_val[target_col])
            self._setup_done = True

            logger.info(
                "HCE setup: train=%d, val=%d, target=%s, task=%s",
                len(df_train_sub), len(df_val), target_col, self._task_type,
            )
            return True

        except Exception as exc:
            logger.warning("HCE: setup failed: %s", exc)
            return False

    def evaluate(self, val_submission_path: Path) -> float | None:
        """Score val_submission.csv against hidden labels.  Higher is always better."""
        if not self._setup_done or self._val_labels is None:
            return None
        if not val_submission_path.exists():
            return None
        try:
            import pandas as pd
            df_pred = pd.read_csv(val_submission_path)
            if df_pred.empty:
                return None

            # Locate prediction column
            pred_col = (
                self._target_col
                if self._target_col and self._target_col in df_pred.columns
                else df_pred.columns[-1]
            )

            # Align by ID when possible
            if self._id_col and self._id_col in df_pred.columns:
                id_to_pred = dict(zip(df_pred[self._id_col], df_pred[pred_col]))
                pairs = [
                    (true, id_to_pred[vid])
                    for true, vid in zip(self._val_labels, self._val_ids)
                    if vid in id_to_pred
                ]
                if not pairs:
                    return None
                y_true, y_pred = zip(*pairs)
            else:
                n = min(len(df_pred), len(self._val_labels))
                y_true = self._val_labels[:n]
                y_pred = df_pred[pred_col].tolist()[:n]

            return self._score(list(y_true), list(y_pred))

        except Exception as exc:
            logger.warning("HCE: evaluation failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _identify_columns(df_sample, df_train) -> tuple[str | None, str | None]:
        """Find (id_col, target_col) by inspecting sample_submission columns."""
        sample_cols = list(df_sample.columns)
        train_cols = set(df_train.columns)

        # ID: first sample column that also appears in train
        id_col = next((c for c in sample_cols if c in train_cols), None)

        # Target: last sample column that appears in train and is not the ID
        target_col = next(
            (c for c in reversed(sample_cols) if c in train_cols and c != id_col),
            None,
        )
        return id_col, target_col

    @staticmethod
    def _detect_task_type(series) -> str:
        try:
            vals = series.dropna()
            if vals.dtype == object or str(vals.dtype).startswith("string"):
                return "classification"
            n_unique = vals.nunique()
            n_total = len(vals)
            if n_unique <= 20 or n_unique / max(n_total, 1) < 0.05:
                return "classification"
            return "regression"
        except Exception:
            return "regression"

    def _score(self, y_true: list, y_pred: list) -> float | None:
        """Compute metric, returning a value where higher is always better."""
        try:
            if self._task_type == "classification":
                return self._classification_score(y_true, y_pred)
            else:
                return self._regression_score(y_true, y_pred)
        except Exception as exc:
            logger.warning("HCE: score computation failed: %s", exc)
            return None

    @staticmethod
    def _classification_score(y_true: list, y_pred: list) -> float | None:
        unique_true = set(str(v) for v in y_true)
        if len(unique_true) == 2:
            try:
                from sklearn.metrics import roc_auc_score
                y_pred_f = [float(p) for p in y_pred]
                true_classes = sorted(unique_true)
                y_true_bin = [1 if str(v) == true_classes[1] else 0 for v in y_true]
                # If predictions are 0/1 labels rather than probabilities → accuracy
                if set(y_pred_f).issubset({0.0, 1.0}):
                    return sum(t == p for t, p in zip(y_true_bin, [int(v) for v in y_pred_f])) / len(y_true_bin)
                return float(roc_auc_score(y_true_bin, y_pred_f))
            except Exception:
                pass
        # Multi-class → accuracy
        return sum(str(t) == str(p) for t, p in zip(y_true, y_pred)) / len(y_true)

    @staticmethod
    def _regression_score(y_true: list, y_pred: list) -> float | None:
        y_true_f = [float(v) for v in y_true]
        y_pred_f = [float(v) for v in y_pred]
        mse = sum((t - p) ** 2 for t, p in zip(y_true_f, y_pred_f)) / len(y_true_f)
        return -(mse ** 0.5)  # negative RMSE — higher is better
