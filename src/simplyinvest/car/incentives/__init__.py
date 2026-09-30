"""The public benefits a vehicle may attract."""

from __future__ import annotations

from .bafa import PREMIUM_2026_BEV, PREMIUM_2026_PHEV, PurchasePremium
from .thg import GhgQuota
from .vehicle_tax import CirculationTaxExemption

__all__ = [
    "PREMIUM_2026_BEV",
    "PREMIUM_2026_PHEV",
    "CirculationTaxExemption",
    "GhgQuota",
    "PurchasePremium",
]
