"""Strategy seeds for the tree search solver.

Each strategy provides a different initial approach hint that biases the first
solution node. Using diverse seeds across parallel attempts is the structural
equivalent of pass@k.
"""

from __future__ import annotations

STRATEGIES: dict[str, str] = {
    "quick_baseline": (
        "Start with the SIMPLEST possible approach. "
        "Read the data, do minimal preprocessing (drop NaNs, label-encode "
        "categoricals), train a single LightGBM or LogisticRegression model "
        "with default hyperparameters, and produce submission.csv as fast as "
        "possible. Prioritize getting a VALID submission before any optimization."
    ),
    "data_first": (
        "Before modeling, spend time understanding the data deeply. "
        "Print column types, missing rates, class balance, value ranges, "
        "and correlations. Identify the most promising features and any "
        "data quality issues. Then build a model that specifically "
        "addresses what you found (e.g. handle class imbalance, engineer "
        "domain-specific features, pick an appropriate model family)."
    ),
    "big_model": (
        "Consider whether this competition benefits from a deep learning "
        "approach. If the data contains images, text, sequences, or signals, "
        "use PyTorch / torchvision / transformers / torchaudio. If the data "
        "is tabular, use a strong gradient boosting model (XGBoost or "
        "CatBoost) with careful feature engineering. Allocate more compute "
        "toward model training rather than quick prototyping."
    ),
    "ensemble_focus": (
        "Build multiple diverse models (e.g. LightGBM, XGBoost, CatBoost, "
        "RandomForest, LogisticRegression) and blend or stack their predictions. "
        "Use out-of-fold predictions for stacking to avoid leakage. Focus on "
        "model diversity over individual model tuning — ensembles of different "
        "algorithms outperform tuning a single one."
    ),
    "feature_heavy": (
        "Invest most effort in feature engineering before modeling. Create "
        "interaction features, polynomial features, aggregations, frequency "
        "encodings, target encodings (with proper CV folds to avoid leakage), "
        "and domain-specific transformations. Use a simple model (LightGBM "
        "with defaults) to let the features do the heavy lifting."
    ),
}

DEFAULT_STRATEGY = "quick_baseline"


def get_strategy(name: str) -> str:
    return STRATEGIES.get(name, STRATEGIES[DEFAULT_STRATEGY])


def all_strategy_names() -> list[str]:
    return list(STRATEGIES.keys())
