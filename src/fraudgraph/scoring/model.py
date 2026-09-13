"""Load the trained model and predict a 0-1 fraud probability (``ml_score``).

The score is an *input* to the deterministic rule engine, never the decider.
If no model has been trained yet, prediction degrades gracefully to a neutral
0.0 so the graph still runs on the rule engine alone.
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

from fraudgraph import config
from fraudgraph.scoring import features as feats


@lru_cache(maxsize=1)
def load_model(model_path: str | None = None):
    """Load and cache the pickled model. Returns None if it doesn't exist."""
    path = Path(model_path) if model_path else config.MODEL_PATH
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


def predict_score(transaction: dict, evidence: dict,
                  model_path: str | None = None) -> float:
    """Predict fraud probability in [0, 1] for a transaction + evidence."""
    model = load_model(model_path)
    if model is None:
        return 0.0
    vector = [feats.to_vector(feats.build_features(transaction, evidence))]
    proba = model.predict_proba(vector)[0][1]
    return float(proba)
