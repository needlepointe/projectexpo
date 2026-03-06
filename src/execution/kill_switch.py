"""
Emergency kill switch — immediately closes all positions and halts trading.
Can be invoked programmatically or via CLI: python -m src.execution.kill_switch
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.execution.alpaca_client import AlpacaClient
from src.database import log_event, init_db

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


def kill(reason: str = "Manual kill switch", confirm: bool = False) -> dict:
    """
    Emergency stop. Closes ALL open positions and cancels ALL open orders.

    Args:
        reason: Why the kill switch was triggered
        confirm: Must be True to execute (safety gate for CLI use)

    Returns:
        dict with positions_closed and orders_cancelled counts
    """
    if not confirm:
        raise RuntimeError("Kill switch requires confirm=True to execute")

    logger.critical("KILL SWITCH ACTIVATED — Reason: %s", reason)
    init_db()
    log_event("kill_switch", reason, "all")

    client = AlpacaClient()

    # Step 1: Cancel all open orders first
    cancelled = client.cancel_all_orders()
    logger.critical("Cancelled %d open orders", len(cancelled))

    # Step 2: Close all positions
    closed = client.close_all_positions()
    logger.critical("Closed %d positions", len(closed))

    # Step 3: Log to file
    log_path = Path("logs") / f"kill_switch_{datetime.now(ET).strftime('%Y%m%d_%H%M%S')}.log"
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"KILL SWITCH TRIGGERED\n")
        f.write(f"Time: {datetime.now(ET).isoformat()}\n")
        f.write(f"Reason: {reason}\n")
        f.write(f"Orders cancelled: {len(cancelled)}\n")
        f.write(f"Positions closed: {len(closed)}\n\n")
        for pos in closed:
            f.write(f"  Closed: {pos}\n")

    result = {
        "triggered_at": datetime.now(ET).isoformat(),
        "reason": reason,
        "orders_cancelled": len(cancelled),
        "positions_closed": len(closed),
        "log_file": str(log_path),
    }
    logger.critical("Kill switch complete: %s", result)
    return result


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Emergency kill switch")
    parser.add_argument("--confirm", action="store_true", help="Required to execute")
    parser.add_argument("--reason", default="CLI kill switch invoked", help="Reason for halt")
    args = parser.parse_args()

    if not args.confirm:
        print("ERROR: You must pass --confirm to execute the kill switch.")
        print("Usage: python -m src.execution.kill_switch --confirm --reason 'Your reason'")
        sys.exit(1)

    result = kill(reason=args.reason, confirm=True)
    print("\n=== KILL SWITCH EXECUTED ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
