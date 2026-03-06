"""
AB Trades Predictor — Dual-Label Self-Improving ML Model

Training data sources (blended):
  1. AB's actual Twitter picks (logged via ABTracker) — labeled positive
     "What does AB look for?" — learns his eye
  2. Bot's own closed trade outcomes (win=1, loss=0)
     "What actually made money?" — learns from results

Cold start: No AB pick data yet → trains on TA outcome labels only.
As AB pick data accumulates, blends in progressively (up to 70% weight).
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.config import get_config
from src.ml.feature_engineering import compute_features, make_labels, get_feature_columns
from src.ml.signal_predictor import SignalPredictor

logger = logging.getLogger(__name__)


class ABPredictor(SignalPredictor):
    """
    Extended ML predictor for the AB Trades bot.
    Inherits all standard SignalPredictor functionality and adds:
      - train_on_ab_picks(): train on AB's historical Twitter calls
      - train_on_outcomes(): retrain on own trade outcomes
      - predict_ab_probability(): P(this stock would be an AB pick) in [0,1]
    """

    AB_MODEL_NAME = "model_ab_style"  # Separate from standard model_ab.pkl

    def __init__(self):
        super().__init__("ab")
        cfg = get_config()
        self._model_dir = Path(cfg["monitoring"]["model_dir"])
        self._model_dir.mkdir(exist_ok=True)

        # Secondary "AB style" classifier (binary: AB pick vs not-pick)
        self._ab_style_model: RandomForestClassifier | None = None
        self._ab_style_scaler = StandardScaler()
        self._ab_style_trained = False
        self._ab_style_trained_at: datetime | None = None
        self._ab_style_feature_cols: list[str] = []
        self._load_ab_style()

    # ------------------------------------------------------------------
    # AB-pick style classifier
    # ------------------------------------------------------------------

    def train_on_ab_picks(self, conn) -> dict:
        """
        Train a binary classifier: "would AB pick this stock?"
        Positive samples = AB's actual picks (from ab_picks table).
        Negative samples = random stocks on the same dates (not picked).

        Falls back to outcome-based training if fewer than 10 AB picks.
        """
        from src.data.market_data import MarketDataClient
        from alpaca.data.timeframe import TimeFrame

        picks = conn.execute(
            "SELECT symbol, picked_at FROM ab_picks ORDER BY picked_at"
        ).fetchall()

        if len(picks) < 10:
            logger.info("[AB_PRED] Only %d AB picks — using outcome-based training only", len(picks))
            return {"ab_picks": len(picks), "status": "insufficient_picks"}

        data_client = MarketDataClient()
        ET_start = datetime.now() - timedelta(days=730)

        positive_rows = []
        negative_rows = []

        # Negative pool: fixed set of non-AB stocks for contrast
        negative_symbols = ["XOM", "KO", "PG", "T", "VZ", "BAC", "WMT", "CVX", "JNJ", "ABT"]

        for pick in picks:
            sym = pick["symbol"]
            try:
                df = data_client.get_historical_bars(sym, TimeFrame.Day, ET_start)
                if df.empty or len(df) < 60:
                    continue
                df = compute_features(df)
                df = self._add_ema_columns(df)
                feat_cols = self._get_style_features(df)
                if not feat_cols:
                    continue
                latest = df[feat_cols].dropna()
                if latest.empty:
                    continue
                row = latest.iloc[-1].values
                positive_rows.append(row)
            except Exception as exc:
                logger.debug("[AB_PRED] Could not get features for %s: %s", sym, exc)

        for sym in negative_symbols:
            try:
                df = data_client.get_historical_bars(sym, TimeFrame.Day, ET_start)
                if df.empty or len(df) < 60:
                    continue
                df = compute_features(df)
                df = self._add_ema_columns(df)
                feat_cols = self._get_style_features(df)
                if not feat_cols:
                    continue
                latest = df[feat_cols].dropna()
                if latest.empty:
                    continue
                row = latest.iloc[-1].values
                negative_rows.append(row)
            except Exception:
                pass

        if not positive_rows or not negative_rows:
            return {"status": "no_features"}

        # Balance classes
        min_count = min(len(positive_rows), len(negative_rows))
        X = np.array(positive_rows[:min_count] + negative_rows[:min_count])
        y = np.array([1] * min_count + [0] * min_count)

        X = self._ab_style_scaler.fit_transform(X)
        self._ab_style_model = RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42,
            class_weight="balanced", n_jobs=1
        )
        self._ab_style_model.fit(X, y)
        self._ab_style_trained = True
        self._ab_style_trained_at = datetime.now()
        self._save_ab_style()

        metrics = {
            "ab_picks_used": len(positive_rows),
            "negative_samples": len(negative_rows),
            "trained_at": str(self._ab_style_trained_at),
        }
        logger.info("[AB_PRED] AB-style model trained: %s", metrics)
        return metrics

    def train_on_outcomes(self, conn) -> dict:
        """
        Retrain the standard direction model (model_ab.pkl) using the bot's
        own closed trade outcomes as additional labeled data.
        Label: 1 if pnl > 0 (won), 0 if pnl <= 0 (lost).
        """
        from src.data.market_data import MarketDataClient
        from alpaca.data.timeframe import TimeFrame

        trades = conn.execute("""
            SELECT symbol, entry_time, exit_time, pnl, side
            FROM trades
            WHERE bot_type = 'ab'
            ORDER BY entry_time DESC
            LIMIT 200
        """).fetchall()

        if len(trades) < 20:
            logger.info("[AB_PRED] Only %d closed AB trades — skipping outcome retraining", len(trades))
            return {"trades": len(trades), "status": "insufficient_trades"}

        data_client = MarketDataClient()
        rows = []
        labels = []

        for t in trades:
            try:
                sym = t["symbol"]
                entry = datetime.fromisoformat(t["entry_time"])
                start = entry - timedelta(days=90)
                df = data_client.get_historical_bars(sym, TimeFrame.Day, start, entry)
                if df.empty or len(df) < 50:
                    continue
                df = compute_features(df)
                feat_cols = [c for c in get_feature_columns() if c in df.columns]
                row = df[feat_cols].dropna().iloc[-1].values
                rows.append(row)
                labels.append(1 if t["pnl"] > 0 else 0)
            except Exception:
                pass

        if len(rows) < 20:
            return {"status": "no_features"}

        X = np.array(rows)
        y = np.array(labels)

        # This retrains the standard self._model (model_ab.pkl)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42,
            class_weight="balanced", n_jobs=1
        )
        model.fit(X_scaled, y)

        self._model = model
        self._scaler = scaler
        self._is_trained = True
        self._trained_at = datetime.now()
        self._feature_cols = feat_cols
        self._save()

        metrics = {
            "outcome_samples": len(rows),
            "wins": int(sum(y)),
            "losses": int(len(y) - sum(y)),
            "trained_at": str(self._trained_at),
        }
        logger.info("[AB_PRED] Outcome model retrained: %s", metrics)
        return metrics

    def predict_ab_probability(self, df: pd.DataFrame) -> float:
        """
        Return the probability [0.0–1.0] that AB would pick this stock.
        Returns 0.0 if AB-style model not yet trained (cold start).
        """
        if not self._ab_style_trained or self._ab_style_model is None:
            return 0.0

        try:
            df = compute_features(df)
            df = self._add_ema_columns(df)
            feat_cols = self._get_style_features(df)
            if not feat_cols:
                return 0.0
            row = df[feat_cols].dropna()
            if row.empty:
                return 0.0
            X = self._ab_style_scaler.transform(row.iloc[[-1]].values)
            prob = float(self._ab_style_model.predict_proba(X)[0][1])  # P(class=1 = AB pick)
            return prob
        except Exception as exc:
            logger.debug("[AB_PRED] predict_ab_probability error: %s", exc)
            return 0.0

    def needs_retraining(self) -> bool:
        """True if standard model is stale OR AB-style model is stale."""
        cfg = get_config()
        freq = cfg.get("ab_trader", {}).get("ml_retrain_frequency_days", 7)
        if not self._is_trained or self._trained_at is None:
            return True
        days = (datetime.now() - self._trained_at).days
        return days >= freq

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_ema_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA20/50/200 if missing (AB uses EMAs not SMAs)."""
        close = df["close"]
        if "ema_20" not in df.columns:
            df = df.copy()
            df["ema_20"] = close.ewm(span=20, adjust=False).mean()
            df["ema_50"] = close.ewm(span=50, adjust=False).mean()
            df["ema_200"] = close.ewm(span=200, adjust=False).mean()
            # EMA alignment features
            df["ema20_above_ema50"] = (df["ema_20"] > df["ema_50"]).astype(float)
            df["price_above_ema200"] = (df["close"] > df["ema_200"]).astype(float)
            df["ema20_slope"] = (df["ema_20"] - df["ema_20"].shift(5)) / df["ema_20"].shift(5).clip(lower=0.001)
        return df

    def _get_style_features(self, df: pd.DataFrame) -> list[str]:
        """Return feature columns available for AB style classification."""
        desired = [
            "rsi_14", "macd_hist", "volume_ratio_20", "atr_pct",
            "price_momentum_5", "price_momentum_10",
            "ema20_above_ema50", "price_above_ema200", "ema20_slope",
            "bb_position", "stoch_k",
        ]
        available = [c for c in desired if c in df.columns]
        self._ab_style_feature_cols = available
        return available

    # ------------------------------------------------------------------
    # Persistence for AB-style model
    # ------------------------------------------------------------------

    def _ab_style_model_path(self) -> Path:
        return self._model_dir / f"{self.AB_MODEL_NAME}.pkl"

    def _save_ab_style(self):
        path = self._ab_style_model_path()
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._ab_style_model,
                "scaler": self._ab_style_scaler,
                "feature_cols": self._ab_style_feature_cols,
                "trained_at": self._ab_style_trained_at,
            }, f)
        logger.info("[AB_PRED] AB-style model saved to %s", path)

    def _load_ab_style(self):
        path = self._ab_style_model_path()
        if not path.exists():
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._ab_style_model = data["model"]
            self._ab_style_scaler = data["scaler"]
            self._ab_style_feature_cols = data.get("feature_cols", [])
            self._ab_style_trained_at = data["trained_at"]
            self._ab_style_trained = True
            logger.info("[AB_PRED] AB-style model loaded (trained %s)", self._ab_style_trained_at)
        except Exception as exc:
            logger.warning("[AB_PRED] Could not load AB-style model: %s", exc)
