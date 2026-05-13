"""Config loader: merges settings.yaml with environment variables and validates."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from privateye.core.exceptions import ConfigError


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    load_dotenv()

    if config_path is None:
        config_path = Path(__file__).parent / "settings.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    # Inject secrets from env vars
    exchanges = cfg.setdefault("exchanges", {})
    binance = exchanges.setdefault("binance", {})
    binance["api_key"] = os.getenv("BINANCE_API_KEY", "")
    binance["api_secret"] = os.getenv("BINANCE_API_SECRET", "")

    alerts = cfg.setdefault("alerts", {})
    alerts["telegram_token"] = os.getenv("TELEGRAM_BOT_TOKEN", "")
    alerts["chat_id"] = os.getenv("TELEGRAM_CHAT_ID", "")
    alerts["smtp_host"] = os.getenv("SMTP_HOST", "smtp.gmail.com")
    alerts["smtp_port"] = int(os.getenv("SMTP_PORT", "587"))
    alerts["smtp_user"] = os.getenv("SMTP_USER", "")
    alerts["smtp_pass"] = os.getenv("SMTP_PASS", "")
    alerts["email_to"] = os.getenv("ALERT_EMAIL_TO", "")

    _validate_config(cfg)
    return cfg


_ALLOWED_MODES = frozenset({
    "backtest", "paper", "live",
    # Phase 6 — robustness modes
    "shadow",
    # Phase 8/10/11 — validation modes
    "walk_forward", "optimize", "kfold",
    # Phase 0 — golden baseline
    "golden_suite",
    # Phase 5 — portfolio backtest
    "portfolio_backtest",
})


def _validate_config(cfg: dict[str, Any]) -> None:
    warnings: list[str] = []

    mode = cfg.get("mode", "paper")
    if mode not in _ALLOWED_MODES:
        allowed = "|".join(sorted(_ALLOWED_MODES))
        raise ConfigError(f"mode must be one of {allowed}, got: {mode!r}")

    if mode == "live":
        # Phase 13 (C-5): inspect every enabled exchange, not hardcoded Binance
        enabled_exchanges = [
            (name, ex) for name, ex in cfg.get("exchanges", {}).items()
            if isinstance(ex, dict) and ex.get("enabled", False)
        ]
        if len(enabled_exchanges) == 0:
            warnings.append("Live mode but no exchange enabled")
        elif len(enabled_exchanges) > 1:
            names = [n for n, _ in enabled_exchanges]
            warnings.append(f"Live mode with multiple enabled exchanges: {names} — only one is supported")
        for name, ex in enabled_exchanges:
            if not ex.get("api_key"):
                warnings.append(f"{name.upper()}_API_KEY not set — live mode will fail")
            if ex.get("sandbox", True):
                warnings.append(f"exchanges.{name}.sandbox=true while mode=live — will be forced to paper mode")

    risk = cfg.get("risk", {})
    if risk.get("max_risk_per_trade_pct", 0.01) > 0.05:
        warnings.append("max_risk_per_trade_pct > 5% — high risk per trade")
    if risk.get("max_daily_drawdown_pct", 0.05) > 0.15:
        warnings.append("max_daily_drawdown_pct > 15% — very loose drawdown limit")

    if not cfg.get("symbols"):
        raise ConfigError("No symbols configured")

    from privateye.utils.logging import get_logger
    log = get_logger()
    for w in warnings:
        log.warning(f"[Config] {w}")


def get_exchange_config(cfg: dict[str, Any], exchange_name: str = "binance") -> dict[str, Any]:
    return cfg["exchanges"][exchange_name]


def get_risk_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("risk", {})


def get_strategy_config(cfg: dict[str, Any], strategy_name: str) -> dict[str, Any]:
    return cfg.get("strategies", {}).get(strategy_name, {})


def get_backtest_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("backtesting", {})
