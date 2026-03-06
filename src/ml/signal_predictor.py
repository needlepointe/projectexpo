"""
ML signal predictor — trains a RandomForest (or GradientBoosting) classifier
on historical data and predicts bullish/bearish/neutral signals.

Usage:
  predictor = SignalPredictor("day")
  predictor.train(historical_df)
  signal, confidence = predictor.predict(recent_df)
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from src.config import get_config
from src.ml.feature_engineering import compute_features, make_labels, get_feature_columns

logger = logging.getLogger(__name__)


class SignalPredictor:
    """
    Hybrid TA + ML signal generator.

    Trains on historical OHLCV data, predicts direction for next N bars.
    Signal: 1 (bullish), -1 (bearish), 0 (neutral)
    """

    def __init__(self, bot_type: str = "day"):
        cfg = get_config()
        self.bot_type = bot_type
        self._ml_cfg = cfg["ml"]
        self._model_dir = Path(cfg["monitoring"]["model_dir"])
        self._model_dir.mkdir(exist_ok=True)

        self._model = None
        self._scaler = StandardScaler()
        self._feature_cols: list[str] = get_feature_columns(self._ml_cfg.get("features"))
        self._is_trained = False
        self._trained_at: datetime | None = None

        # Try loading a saved model
        self._load()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> dict:
        """
        Train model on historical OHLCV DataFrame.
        Returns training metrics dict.
        """
        cfg = self._ml_cfg
        min_samples = cfg.get("min_training_samples", 200)

        df = compute_features(df)
        labels = make_labels(
            df,
            forward_bars=cfg.get("forward_return_bars", 6),
            bullish_thr=cfg.get("bullish_threshold", 0.002),
            bearish_thr=cfg.get("bearish_threshold", -0.002),
            use_percentile=True,   # Guarantees ~33% bull / 33% bear / 33% neutral
            percentile=0.33,
        )

        df = df.copy()
        df["label"] = labels

        # Drop rows with NaN features or labels
        feature_cols = [c for c in self._feature_cols if c in df.columns]
        df = df.dropna(subset=feature_cols + ["label"])

        if len(df) < min_samples:
            raise ValueError(
                f"Not enough training samples: {len(df)} < {min_samples}. "
                "Fetch more historical data."
            )

        X = df[feature_cols].values
        y = df["label"].values

        X = self._scaler.fit_transform(X)

        model_type = cfg.get("model_type", "random_forest")
        if model_type == "gradient_boosting":
            self._model = GradientBoostingClassifier(
                n_estimators=cfg.get("n_estimators", 200),
                max_depth=cfg.get("max_depth", 5),
                random_state=42,
            )
        else:
            self._model = RandomForestClassifier(
                n_estimators=cfg.get("n_estimators", 200),
                max_depth=cfg.get("max_depth", 8),
                min_samples_split=10,
                n_jobs=1,   # 1 = no joblib workers; avoids massive overhead on repeated single-row calls
                random_state=42,
                class_weight="balanced",
            )

        # Time-series cross-validation (no data leakage)
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            self._model.fit(X[train_idx], y[train_idx])
            score = self._model.score(X[val_idx], y[val_idx])
            cv_scores.append(score)

        # Final fit on all data
        self._model.fit(X, y)
        self._feature_cols = feature_cols
        self._is_trained = True
        self._trained_at = datetime.now()
        self._save()

        metrics = {
            "samples": len(df),
            "features": len(feature_cols),
            "cv_accuracy_mean": float(np.mean(cv_scores)),
            "cv_accuracy_std": float(np.std(cv_scores)),
            "label_distribution": {
                "bullish": int((y == 1).sum()),
                "bearish": int((y == -1).sum()),
                "neutral": int((y == 0).sum()),
            },
            "trained_at": str(self._trained_at),
        }
        logger.info("[%s] ML model trained: %s", self.bot_type.upper(), metrics)
        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> tuple[int, float]:
        """
        Predict signal for the most recent bar.

        Returns: (signal: int, confidence: float)
          signal: 1 (bullish), -1 (bearish), 0 (neutral)
          confidence: probability of predicted class (0–1)
        """
        if not self._is_trained or self._model is None:
            logger.warning("ML model not trained — returning neutral signal")
            return 0, 0.0

        df = compute_features(df)
        feature_cols = [c for c in self._feature_cols if c in df.columns]
        if not feature_cols:
            return 0, 0.0

        recent = df[feature_cols].dropna()
        if recent.empty:
            return 0, 0.0

        X = self._scaler.transform(recent.iloc[[-1]].values)
        signal = int(self._model.predict(X)[0])
        proba = self._model.predict_proba(X)[0]
        confidence = float(max(proba))

        min_conf = self._ml_cfg.get("min_confidence", 0.55)
        if confidence < min_conf:
            signal = 0  # Below confidence threshold → neutral

        logger.debug(
            "[%s] ML signal: %s (confidence %.1f%%)",
            self.bot_type.upper(), _signal_name(signal), confidence * 100,
        )
        return signal, confidence

    def predict_row(self, row: pd.Series) -> tuple[int, float]:
        """
        Predict signal from a single pre-computed feature row (no feature recomputation).
        Use this in backtesting where features are already computed on the full DataFrame.
        """
        if not self._is_trained or self._model is None:
            return 0, 0.0

        feature_cols = [c for c in self._feature_cols if c in row.index]
        if not feature_cols:
            return 0, 0.0

        values = row[feature_cols].values
        if any(v != v for v in values):  # fast NaN check
            return 0, 0.0

        X = self._scaler.transform([values])
        signal = int(self._model.predict(X)[0])
        proba = self._model.predict_proba(X)[0]
        confidence = float(max(proba))

        if confidence < self._ml_cfg.get("min_confidence", 0.55):
            signal = 0
        return signal, confidence

    def needs_retraining(self) -> bool:
        """Return True if model hasn't been trained or is overdue for retraining."""
        if not self._is_trained or self._trained_at is None:
            return True
        days_since = (datetime.now() - self._trained_at).days
        return days_since >= self._ml_cfg.get("retrain_frequency_days", 7)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _model_path(self) -> Path:
        return self._model_dir / f"model_{self.bot_type}.pkl"

    def _save(self):
        path = self._model_path()
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "scaler": self._scaler,
                "feature_cols": self._feature_cols,
                "trained_at": self._trained_at,
            }, f)
        logger.info("Model saved to %s", path)

    def _load(self):
        path = self._model_path()
        if not path.exists():
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._model = data["model"]
            self._scaler = data["scaler"]
            self._feature_cols = data["feature_cols"]
            self._trained_at = data["trained_at"]
            self._is_trained = True
            logger.info("Model loaded from %s (trained %s)", path, self._trained_at)
        except Exception as exc:
            logger.warning("Could not load model from %s: %s", path, exc)


def _signal_name(signal: int) -> str:
    return {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}.get(signal, "UNKNOWN")
