"""A battery vehicle, and the losses between the meter and the cells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from simplyinvest.car.price import DEFAULT_ESCALATION, ChargingTariff, EnergyPrice
from simplyinvest.money import Quantity

from .base import MeteredSource

__all__ = ["ELECTRICITY", "Electricity"]

#: The carrier a battery vehicle draws on.
ELECTRICITY = "electricity"


@dataclass(frozen=True)
class Electricity(MeteredSource):
    """Grid electricity, consumed at the battery and billed at the meter.

    ``consumption`` is measured at the battery and ``charging_loss`` is taken
    between meter and battery, so the billed energy is the larger figure.
    Where the car is charged, and at what price, is quoted on the buyer.

    Args:
        charging_loss: Fraction of energy lost between meter and battery.

    Raises:
        ValueError: if the charging loss is outside ``[0, 1)``.
    """

    charging_loss: float = 0.0

    unit: ClassVar[str] = "kWh"
    carrier: ClassVar[str] = ELECTRICITY

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.charging_loss < 1.0:
            raise ValueError(f"charging_loss is a fraction below one, got {self.charging_loss!r}")

    @classmethod
    def price(
        cls,
        price: Quantity = 0.0,
        *,
        public: Quantity | None = None,
        home_share: float = 1.0,
        escalation: str | float = DEFAULT_ESCALATION,
    ) -> EnergyPrice:
        """A charging contract, blended between home and public.

        Args:
            price: What one kWh costs at home.
            public: What one kWh costs in public.  Defaults to the home price.
            home_share: Fraction of energy drawn at home.
            escalation: Escalation key or annual rate the prices grow at.
        """
        return ChargingTariff(
            home=price,
            public=price if public is None else public,
            home_share=home_share,
            escalation=escalation,
            carrier=cls.carrier,
        )

    @property
    def effective_consumption(self) -> Quantity:
        """Energy drawn at the meter, per 100 km."""
        return self.consumption * self.real_world_factor / (1.0 - self.charging_loss)
