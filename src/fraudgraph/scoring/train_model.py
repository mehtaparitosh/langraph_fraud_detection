"""Train the tiny local fraud model offline and save it to ``model.pkl``.

Builds features for every seeded transaction using the exact same
``build_features`` path used at inference (via ``db.gather_evidence``), so
train-time and predict-time features can never drift. Prefers XGBoost; falls
back to scikit-learn's LogisticRegression if XGBoost isn't importable.

Run:  ``uv run python -m fraudgraph.scoring.train_model``
"""

from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

from fraudgraph import config
from fraudgraph.data import db, seed
from fraudgraph.scoring import features as feats


def _load_rows(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(
            "SELECT txn_id, card_id, device_id, ip, merchant_id, amount, ts, geo, label "
            "FROM transactions"
        )]
    finally:
        conn.close()


def build_dataset(db_path: Path | None = None):
    """Return (X, y): feature vectors and labels for every seeded transaction."""
    db_path = db_path or config.DB_PATH
    if not db_path.exists():
        seed.seed(db_path=db_path)

    X, y = [], []
    for row in _load_rows(db_path):
        evidence = db.gather_evidence(row, db_path=db_path)
        X.append(feats.to_vector(feats.build_features(row, evidence)))
        y.append(int(row["label"]))
    return X, y


def _new_classifier():
    """XGBoost if available, else logistic regression. Both give predict_proba."""
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=42,
        ), "xgboost"
    except Exception:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000, class_weight="balanced"), "logistic"


def train(db_path: Path | None = None, model_path: Path | None = None) -> Path:
    """Train on the seeded data, report a held-out AUC, and pickle the model."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    model_path = model_path or config.MODEL_PATH
    X, y = build_dataset(db_path)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model, kind = _new_classifier()
    model.fit(X_tr, y_tr)

    proba = [p[1] for p in model.predict_proba(X_te)]
    auc = roc_auc_score(y_te, proba)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump(model, fh)

    try:
        shown = model_path.relative_to(config.PROJECT_ROOT)
    except ValueError:
        shown = model_path

    print(f"Trained {kind} on {len(X)} rows "
          f"({sum(y)} fraud / {len(y) - sum(y)} legit)")
    print(f"  features : {feats.FEATURE_NAMES}")
    print(f"  test AUC : {auc:.3f}")
    print(f"  saved    : {shown}")
    return model_path


def main() -> None:
    train()


if __name__ == "__main__":
    main()
