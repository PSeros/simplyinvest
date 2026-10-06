"""German benefits attaching to a photovoltaic system."""

from __future__ import annotations

from .eeg import (
    EEG_2026_AUGUST,
    EEG_2027_DRAFT,
    ENACTED_REGIMES,
    Band,
    FeedInTariff,
    FeedInValue,
    FixedTariff,
    Regime,
    blended_rate,
)
from .vat import STANDARD_VAT_RATE, ZeroRatedSupply

__all__ = [
    "EEG_2026_AUGUST",
    "EEG_2027_DRAFT",
    "ENACTED_REGIMES",
    "STANDARD_VAT_RATE",
    "Band",
    "FeedInTariff",
    "FeedInValue",
    "FixedTariff",
    "Regime",
    "ZeroRatedSupply",
    "blended_rate",
]
