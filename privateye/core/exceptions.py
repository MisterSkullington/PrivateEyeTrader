"""Domain exceptions for PrivateEyeTrader."""


class PrivateEyeError(Exception):
    """Base exception."""


class InsufficientFundsError(PrivateEyeError):
    """Not enough cash to open the requested position."""


class RiskLimitBreachedError(PrivateEyeError):
    """Order rejected because it would violate a risk limit."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class DailyDrawdownBreachedError(PrivateEyeError):
    """Daily drawdown circuit breaker triggered — trading halted."""


class ExchangeError(PrivateEyeError):
    """Wraps exchange API errors."""


class DataError(PrivateEyeError):
    """Data fetch, parse, or alignment error."""


class ConfigError(PrivateEyeError):
    """Invalid or missing configuration."""


class KillSwitchActivatedError(PrivateEyeError):
    """Kill switch was triggered — all positions closing, trading halted."""
