"""Central configuration loader. Single source of truth for all settings."""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_CONFIG: dict | None = None
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"


def get_config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        with open(_CONFIG_PATH) as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG


def reload_config() -> dict:
    """Force reload from disk (useful after editing settings.yaml)."""
    global _CONFIG
    _CONFIG = None
    return get_config()


def get_alpaca_credentials() -> tuple[str, str]:
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        raise EnvironmentError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")
    return api_key, secret_key


def is_paper_trading() -> bool:
    return get_config()["account"]["paper_trading"]


def is_options_enabled() -> bool:
    return get_config()["options"]["enabled"]


def is_ml_enabled() -> bool:
    return get_config()["ml"]["enabled"]
