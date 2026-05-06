"""
Sanctions and known-bad-actor symbol checker.

Maintains a curated set of symbols subject to regulatory action, OFAC
sanctions, or proven fraud.  Every trade proposal is checked against this
list regardless of other compliance settings.

Users may extend the list via ``compliance.sanctions.extra_banned`` in
``settings.yaml`` without touching code.

Sources:
  LUNA/LUNC/UST  — Terra collapse; OFAC action; class-action fraud (2022)
  FTT            — FTX customer fraud, SEC/DOJ charges (2022)
  TORN           — Tornado Cash; OFAC SDN designation (Aug 2022)
  SQUID          — Squid Game token; 100 % rug-pull (Nov 2021)
"""
from __future__ import annotations

# Immutable built-in sanctions list (upper-cased for comparison).
_KNOWN_SANCTIONED: frozenset[str] = frozenset({
    # Terra / Luna
    "LUNA/USDT", "LUNA/BTC", "LUNA/USD",
    "LUNC/USDT", "LUNC/BTC",
    "UST/USDT",  "UST/USD",
    # FTX token
    "FTT/USDT",  "FTT/BTC",  "FTT/USD",
    # Tornado Cash (OFAC SDN)
    "TORN/USDT", "TORN/ETH", "TORN/BTC",
    # Squid Game rug-pull
    "SQUID/USDT", "SQUID/BNB",
})


class SanctionsChecker:
    """Check whether a symbol is on the sanctions / known-bad-actor list."""

    def __init__(
        self,
        enabled:      bool        = True,
        extra_banned: list[str]   | None = None,
    ) -> None:
        self._enabled = enabled
        extras = {s.upper() for s in (extra_banned or [])}
        self._banned: frozenset[str] = (
            frozenset({s.upper() for s in _KNOWN_SANCTIONED}) | extras
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "SanctionsChecker":
        """Build from the ``compliance.sanctions`` settings dict."""
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            extra_banned=list(cfg.get("extra_banned", [])),
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def check(self, symbol: str) -> tuple[bool, str]:
        """
        Returns ``(allowed: bool, reason: str)``.

        * ``allowed=True``  — symbol not sanctioned.
        * ``allowed=False`` — symbol blocked.
        """
        if not self._enabled:
            return True, ""
        if symbol.upper() in self._banned:
            return (
                False,
                f"Symbol {symbol} is on the sanctions / known-bad-actor list",
            )
        return True, ""
