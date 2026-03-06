"""
Unit tests for strategy signal generation and feature engineering.
"""

import pytest
import numpy as np
import pandas as pd

from src.ml.feature_engineering import compute_features, make_labels, _rsi
from src.strategies.base_strategy import TradeSignal


def make_ohlcv(n: int = 100, trend: str = "up") -> pd.DataFrame:
    """Generate synthetic OHLCV bars for testing."""
    np.random.seed(42)
    base = 100.0
    prices = [base]
    for _ in range(n - 1):
        change = np.random.normal(0.001 if trend == "up" else -0.001, 0.01)
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    high = prices * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.005, n)))
    volume = np.random.randint(100000, 2000000, n)

    df = pd.DataFrame({
        "open": prices * 0.999,
        "high": high,
        "low": low,
        "close": prices,
        "volume": volume.astype(float),
        "vwap": prices * 1.001,
    })
    df.index = pd.date_range("2024-01-01", periods=n, freq="5min")
    return df


class TestFeatureEngineering:
    def test_compute_features_adds_rsi(self):
        df = make_ohlcv(100)
        result = compute_features(df)
        assert "rsi_14" in result.columns
        # RSI should be 0–100
        valid = result["rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_compute_features_adds_macd(self):
        df = make_ohlcv(100)
        result = compute_features(df)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns

    def test_compute_features_adds_bb(self):
        df = make_ohlcv(100)
        result = compute_features(df)
        assert "bb_position" in result.columns

    def test_compute_features_adds_volume_ratio(self):
        df = make_ohlcv(100)
        result = compute_features(df)
        assert "volume_ratio_20" in result.columns

    def test_compute_features_adds_vwap_deviation(self):
        df = make_ohlcv(100)
        result = compute_features(df)
        assert "vwap_deviation" in result.columns

    def test_compute_features_empty_df_returns_empty(self):
        df = pd.DataFrame()
        result = compute_features(df)
        assert result.empty

    def test_compute_features_short_df_returns_unchanged(self):
        df = make_ohlcv(5)
        result = compute_features(df)
        # Should return without crashing (may not have all features)
        assert len(result) == 5

    def test_rsi_values_in_range(self):
        series = pd.Series([100.0 + i * 0.5 for i in range(50)])
        rsi = _rsi(series, 14)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_make_labels_generates_classes(self):
        df = make_ohlcv(200)
        labels = make_labels(df, forward_bars=6)
        assert set(labels.unique()).issubset({-1, 0, 1})
        # Should have some of each class
        assert (labels == 1).sum() > 0
        assert (labels == -1).sum() > 0

    def test_momentum_features_present(self):
        df = make_ohlcv(100)
        result = compute_features(df)
        assert "price_momentum_5" in result.columns
        assert "price_momentum_10" in result.columns

    def test_moving_averages_present(self):
        df = make_ohlcv(220)
        result = compute_features(df)
        assert "ma_20" in result.columns
        assert "ma_50" in result.columns
        assert "ma_200" in result.columns


class TestTradeSignal:
    def test_signal_rr_ratio(self):
        signal = TradeSignal(
            symbol="AAPL", direction="long",
            entry_price=100.0, stop_loss=98.0, take_profit=104.0,
            confidence=0.7, reason="test",
            ml_signal=1, ml_confidence=0.7, ta_signal=1, timeframe="5min",
        )
        assert signal.rr_ratio() == pytest.approx(2.0)

    def test_signal_is_long(self):
        signal = TradeSignal(
            symbol="AAPL", direction="long",
            entry_price=100.0, stop_loss=98.0, take_profit=104.0,
            confidence=0.7, reason="test",
            ml_signal=1, ml_confidence=0.7, ta_signal=1, timeframe="5min",
        )
        assert signal.is_long()

    def test_signal_risk_distance(self):
        signal = TradeSignal(
            symbol="AAPL", direction="long",
            entry_price=100.0, stop_loss=98.0, take_profit=104.0,
            confidence=0.7, reason="test",
            ml_signal=1, ml_confidence=0.7, ta_signal=1, timeframe="5min",
        )
        assert signal.risk_distance() == pytest.approx(2.0)

    def test_signal_reward_distance(self):
        signal = TradeSignal(
            symbol="AAPL", direction="long",
            entry_price=100.0, stop_loss=98.0, take_profit=104.0,
            confidence=0.7, reason="test",
            ml_signal=1, ml_confidence=0.7, ta_signal=1, timeframe="5min",
        )
        assert signal.reward_distance() == pytest.approx(4.0)


class TestMLSignalPredictor:
    def test_predict_returns_neutral_without_training(self):
        """Untrained model should return neutral signal."""
        from unittest.mock import patch, MagicMock
        import pickle

        cfg = {
            "ml": {
                "model_type": "random_forest",
                "n_estimators": 10,
                "max_depth": 3,
                "retrain_frequency_days": 7,
                "training_lookback_days": 365,
                "min_training_samples": 50,
                "min_confidence": 0.55,
                "forward_return_bars": 6,
                "bullish_threshold": 0.008,
                "bearish_threshold": -0.008,
                "features": None,
            },
            "monitoring": {"model_dir": "/tmp/test_models"},
        }

        with patch("src.ml.signal_predictor.get_config", return_value=cfg):
            with patch("src.ml.signal_predictor.Path") as mock_path:
                mock_path.return_value.exists.return_value = False
                mock_path.return_value.__truediv__ = lambda self, x: self
                mock_path.return_value.mkdir = MagicMock()
                from src.ml.signal_predictor import SignalPredictor
                predictor = SignalPredictor.__new__(SignalPredictor)
                predictor._is_trained = False
                predictor._model = None
                signal, conf = predictor.predict(make_ohlcv(100))
        assert signal == 0
        assert conf == 0.0
