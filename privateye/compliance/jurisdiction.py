"""
Jurisdiction-aware symbol filtering.

When a user sets ``compliance.jurisdiction`` to an ISO 3166-1 alpha-2
country code, trading signals for symbols restricted in that jurisdiction
are suppressed before reaching the risk manager.

Rules are intentionally conservative and fail-open: unrecognised
jurisdictions allow all trading.  Users must opt-in by setting a code.

Sources:
  CN — People's Bank of China circular, Sep 2021
  US — SEC enforcement actions (Ripple, LBRY)
  EG — Central Bank of Egypt directive, 2018
  MA — Bank Al-Maghrib communiqué, 2017
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Maps ISO 3166-1 alpha-2 → trading restrictions.
# "block_all" takes precedence over "blocked_symbols".
JURISDICTION_RULES: dict[str, dict] = {
    "CN": {
        "block_all": True,
        "blocked_symbols": [],
        "reason": "All crypto trading prohibited (PBoC, Sep 2021)",
    },
    "US": {
        "block_all": False,
        "blocked_symbols": [
            "XRP/USDT", "XRP/BTC", "XRP/USD",   # SEC v. Ripple (2020–2023)
            "LBRY/BTC", "LBRY/USDT",             # SEC v. LBRY (2021)
        ],
        "reason": "SEC enforcement action — potential unregistered securities",
    },
    "EG": {
        "block_all": True,
        "blocked_symbols": [],
        "reason": "Crypto transactions prohibited (Central Bank of Egypt, 2018)",
    },
    "MA": {
        "block_all": True,
        "blocked_symbols": [],
        "reason": "Crypto transactions prohibited (Bank Al-Maghrib, 2017)",
    },
}


@dataclass
class JurisdictionFilter:
    """Block symbols restricted under the user's declared jurisdiction."""

    jurisdiction: str = ""   # ISO 3166-1 alpha-2; empty string = disabled

    @classmethod
    def from_config(cls, cfg: dict) -> "JurisdictionFilter":
        """Build from the ``compliance:`` settings dict."""
        return cls(jurisdiction=str(cfg.get("jurisdiction", "")).upper())

    @property
    def enabled(self) -> bool:
        """True when a non-empty jurisdiction code is set."""
        return bool(self.jurisdiction)

    def check(self, symbol: str) -> tuple[bool, str]:
        """
        Check whether *symbol* may be traded in the configured jurisdiction.

        Returns ``(allowed: bool, reason: str)``.

        * ``allowed=True``  — trading permitted.
        * ``allowed=False`` — trading blocked; *reason* explains why.

        Unknown jurisdictions always return ``(True, "")`` (fail-open).
        """
        if not self.enabled:
            return True, ""

        rules = JURISDICTION_RULES.get(self.jurisdiction)
        if rules is None:
            return True, ""   # unrecognised jurisdiction → fail-open

        if rules.get("block_all", False):
            return False, f"[{self.jurisdiction}] {rules['reason']}"

        blocked = {s.upper() for s in rules.get("blocked_symbols", [])}
        if symbol.upper() in blocked:
            return False, (
                f"[{self.jurisdiction}] {rules['reason']}: {symbol}"
            )

        return True, ""
