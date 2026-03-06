"""
AB Tracker — Monitor @ABTradess for stock picks and track prediction accuracy.

Two modes:
  1. AUTO mode  (requires TWITTER_BEARER_TOKEN in .env):
     Automatically scans AB's recent tweets for $TICKER mentions every hour.

  2. MANUAL mode (no Twitter API needed):
     User logs AB's picks via: python main.py --log-ab-pick SYMBOL
     Or programmatically: tracker.log_ab_pick("NVDA")

After a pick is logged, the tracker:
  - Checks if the AB bot had pre-called this symbol in ab_predictions
  - Updates ab_predictions.ab_picked = 1 for any recent matching prediction
  - Logs the hit/miss for accuracy tracking

Weekly accuracy report shows: "Bot predicted X of Y AB picks before he called them"
"""

import logging
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.config import get_config
from src.database import get_connection, log_event

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

# US stock ticker pattern: $AAPL or standalone uppercase 2-5 letter word
TICKER_PATTERN = re.compile(r"\$([A-Z]{1,5})\b|(?<!\w)([A-Z]{2,5})(?!\w)")

# Known non-ticker uppercase words to filter out
COMMON_WORDS = {
    "I", "A", "THE", "IN", "ON", "AT", "BY", "UP", "DO", "SO", "GO", "MY", "BE",
    "IT", "TO", "OF", "IS", "AS", "AM", "OR", "AND", "BUT", "FOR", "NOT", "YOU",
    "ARE", "HAS", "HAD", "WAS", "GET", "GOT", "LET", "PUT", "SET", "RUN", "MAY",
    "CAN", "ETF", "IPO", "ATH", "ATL", "DCA", "PnL", "USD", "PDT", "PDT", "EOD",
    "RSI", "EMA", "SMA", "MACD", "VWAP", "BOS", "TP", "SL", "RR", "WR", "US",
    "NYC", "CEO", "CFO", "COO", "PM", "AM", "ET", "EST", "PST", "CST", "MST",
    "EOW", "EOY", "YTD", "QOQ", "YOY", "FOMO", "FUD", "YOLO", "GMO",
}


class ABTracker:
    """
    Tracks @ABTradess picks and measures the bot's predictive accuracy.
    """

    def __init__(self):
        self._cfg = get_config().get("ab_trader", {})
        self._twitter_handle = self._cfg.get("twitter_handle", "ABTradess")
        self._twitter_enabled = self._cfg.get("twitter_monitor", True)
        self._twitter_client = None

        # Try to initialize Twitter client
        if self._twitter_enabled:
            self._init_twitter()

    def _init_twitter(self):
        """Initialize Tweepy client if TWITTER_BEARER_TOKEN is in environment."""
        import os
        bearer_token = os.getenv("TWITTER_BEARER_TOKEN", "")
        if not bearer_token:
            logger.info(
                "[AB_TRACKER] No TWITTER_BEARER_TOKEN found — running in MANUAL mode. "
                "Log AB's picks with: python main.py --log-ab-pick SYMBOL"
            )
            return
        try:
            import tweepy
            self._twitter_client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
            logger.info("[AB_TRACKER] Twitter/X client initialized — AUTO mode active")
        except ImportError:
            logger.warning(
                "[AB_TRACKER] tweepy not installed. Run: pip install tweepy. "
                "Falling back to manual mode."
            )
        except Exception as exc:
            logger.warning("[AB_TRACKER] Twitter init failed: %s — manual mode", exc)

    # ------------------------------------------------------------------
    # Tweet scanning
    # ------------------------------------------------------------------

    def scan_recent_tweets(self, lookback_hours: int = 24) -> list[str]:
        """
        Scan @ABTradess tweets for stock ticker mentions.
        Returns list of ticker symbols found. Empty list if no API or no picks.
        """
        if self._twitter_client is None:
            logger.debug("[AB_TRACKER] Manual mode — no tweet scanning")
            return []

        try:
            import tweepy
            # Look up user ID for the handle
            user = self._twitter_client.get_user(username=self._twitter_handle)
            if not user or not user.data:
                return []
            user_id = user.data.id

            # Fetch recent tweets
            since = datetime.now(ET) - timedelta(hours=lookback_hours)
            tweets = self._twitter_client.get_users_tweets(
                id=user_id,
                start_time=since.isoformat(),
                max_results=100,
                tweet_fields=["created_at", "text"],
            )
            if not tweets or not tweets.data:
                return []

            tickers_found = set()
            for tweet in tweets.data:
                found = self._extract_tickers(tweet.text)
                for sym in found:
                    tickers_found.add(sym)
                    self.log_ab_pick(sym, tweet_text=tweet.text[:280])

            result = list(tickers_found)
            if result:
                logger.info("[AB_TRACKER] Found %d tickers in AB's tweets: %s", len(result), result)
            return result

        except Exception as exc:
            logger.warning("[AB_TRACKER] Tweet scan failed: %s", exc)
            return []

    def _extract_tickers(self, text: str) -> list[str]:
        """Extract stock ticker symbols from tweet text."""
        tickers = set()
        matches = TICKER_PATTERN.findall(text)
        for dollar_sym, plain_sym in matches:
            sym = (dollar_sym or plain_sym).upper()
            if sym and sym not in COMMON_WORDS and len(sym) >= 2:
                tickers.add(sym)
        return list(tickers)

    # ------------------------------------------------------------------
    # Manual pick logging
    # ------------------------------------------------------------------

    def log_ab_pick(self, symbol: str, tweet_text: str = "", auto_detected: bool = False) -> dict:
        """
        Log a stock pick by AB. Called either automatically (from tweet scan)
        or manually by the user (via CLI).

        Returns: dict with was_predicted, bot_confidence, matching_prediction_id
        """
        symbol = symbol.upper().strip()
        now = datetime.now(ET).isoformat()

        # Find any pre-call prediction we made for this symbol in the last 7 days
        lookback = (datetime.now(ET) - timedelta(days=7)).isoformat()
        with get_connection() as conn:
            pred = conn.execute("""
                SELECT id, confidence FROM ab_predictions
                WHERE symbol = ? AND predicted_at >= ? AND ab_picked = 0
                ORDER BY predicted_at DESC
                LIMIT 1
            """, (symbol, lookback)).fetchone()

            was_predicted = 1 if pred else 0
            bot_confidence = float(pred["confidence"]) if pred else 0.0

            # Log the pick
            conn.execute("""
                INSERT INTO ab_picks (symbol, picked_at, tweet_text, was_predicted, bot_confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (symbol, now, tweet_text, was_predicted, bot_confidence))

            # Mark the matching prediction as hit
            if pred:
                conn.execute(
                    "UPDATE ab_predictions SET ab_picked = 1 WHERE id = ?",
                    (pred["id"],)
                )
                logger.info(
                    "[AB_TRACKER] HIT! Bot predicted %s (conf=%.1f%%) %s before AB called it",
                    symbol, bot_confidence * 100,
                    "automatically" if auto_detected else "manually",
                )
            else:
                logger.info(
                    "[AB_TRACKER] AB picked %s — bot had no pre-call prediction",
                    symbol,
                )

        source = "auto" if auto_detected else "manual"
        log_event("ab_pick", f"AB picked {symbol} [{source}], was_predicted={was_predicted}", "ab")

        return {
            "symbol": symbol,
            "was_predicted": bool(was_predicted),
            "bot_confidence": bot_confidence,
        }

    # ------------------------------------------------------------------
    # Prediction logging
    # ------------------------------------------------------------------

    def log_bot_predictions(self, signals: list) -> None:
        """
        Log the bot's top predictions to ab_predictions for later comparison
        against AB's actual picks.

        Args:
            signals: list of TradeSignal objects (top picks from EOD scan)
        """
        if not signals:
            return
        now = datetime.now(ET).isoformat()
        with get_connection() as conn:
            for signal in signals:
                conn.execute("""
                    INSERT INTO ab_predictions (symbol, predicted_at, confidence, signal_reason)
                    VALUES (?, ?, ?, ?)
                """, (
                    signal.symbol,
                    now,
                    signal.confidence,
                    getattr(signal, "reason", ""),
                ))
        logger.info(
            "[AB_TRACKER] Logged %d bot predictions to ab_predictions table",
            len(signals),
        )

    # ------------------------------------------------------------------
    # Accuracy reporting
    # ------------------------------------------------------------------

    def get_accuracy_report(self) -> dict:
        """
        How often did the bot predict AB's picks before he called them?
        Compares ab_picks vs ab_predictions within ±7 days.
        """
        with get_connection() as conn:
            total_picks = conn.execute(
                "SELECT COUNT(*) as c FROM ab_picks"
            ).fetchone()["c"]

            predicted_before = conn.execute(
                "SELECT COUNT(*) as c FROM ab_picks WHERE was_predicted = 1"
            ).fetchone()["c"]

            recent_predictions = conn.execute("""
                SELECT COUNT(*) as c FROM ab_predictions
                WHERE predicted_at >= datetime('now', '-30 days')
            """).fetchone()["c"]

            top_hits = conn.execute("""
                SELECT p.symbol, p.picked_at, p.bot_confidence
                FROM ab_picks p
                WHERE p.was_predicted = 1
                ORDER BY p.picked_at DESC
                LIMIT 10
            """).fetchall()

        hit_rate = (predicted_before / total_picks * 100) if total_picks > 0 else 0

        return {
            "total_ab_picks": total_picks,
            "predicted_before": predicted_before,
            "hit_rate_pct": hit_rate,
            "recent_predictions_30d": recent_predictions,
            "top_hits": [
                {"symbol": r["symbol"], "date": r["picked_at"], "confidence": r["bot_confidence"]}
                for r in top_hits
            ],
        }

    def print_accuracy_report(self) -> None:
        """Print a human-readable accuracy report to console."""
        report = self.get_accuracy_report()
        print("\n=== AB TRACKER ACCURACY REPORT ===")
        print(f"  Total AB picks logged:    {report['total_ab_picks']}")
        print(f"  Bot predicted before AB:  {report['predicted_before']}")
        print(f"  Hit rate:                 {report['hit_rate_pct']:.1f}%")
        print(f"  Predictions (last 30d):   {report['recent_predictions_30d']}")
        if report["top_hits"]:
            print("\n  Recent hits:")
            for h in report["top_hits"]:
                print(f"    {h['symbol']:8s} @ {h['date'][:10]}  conf={h['confidence']:.1%}")
        print("================================\n")
