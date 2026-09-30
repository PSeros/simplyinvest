"""The asset contract a domain package implements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from simplyinvest.money import Amount
    from simplyinvest.timeline import Term

__all__ = ["Asset"]


@runtime_checkable
class Asset(Protocol):
    """Whatever is being invested in."""

    @property
    def name(self) -> str:
        """What to call it in a report."""
        ...

    @property
    def capital_cost(self) -> Amount:
        """The acquisition outlay."""
        ...

    @property
    def setup_cost(self) -> Amount:
        """Outlay at commissioning that is not part of the price."""
        ...

    @property
    def economic_life(self) -> Term | None:
        """How long it remains useful, or ``None`` if it outlives any horizon."""
        ...

    @property
    def age_at_acquisition(self) -> Term:
        """How old it already is when acquired."""
        ...

    def residual_value(self, *, held: Term) -> Amount:
        """What it is worth after being held for ``held``."""
        ...
