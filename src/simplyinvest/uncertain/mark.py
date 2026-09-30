"""Declaring uncertainty where the number is written."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from simplyinvest.errors import ParameterError

if TYPE_CHECKING:
    from simplyinvest.money import Quantity

    from .distribution import Distribution

__all__ = ["Uncertain", "uncertain"]


class Uncertain(float):
    """A float carrying the label and prior it was declared with."""

    __slots__ = ("label", "prior")

    label: str
    prior: Distribution | None

    def __new__(cls, base: float, label: str, prior: Distribution | None = None) -> Uncertain:
        """A float equal to ``base`` that remembers how uncertain it is."""
        marked = super().__new__(cls, base)
        marked.label = label
        marked.prior = prior
        return marked

    def __getnewargs__(self) -> tuple[Any, ...]:
        return (float(self), self.label, self.prior)

    def __repr__(self) -> str:
        return f"uncertain({float(self)!r}, {self.label!r})"


def uncertain(base: float, label: str, prior: Distribution | None = None) -> Quantity:
    """``base``, marked as uncertain and addressable by ``label``.

    Args:
        base: The value every deterministic appraisal uses.
        label: What to call this parameter, unique within one model.
        prior: How it is distributed.  A simulation needs one, from here or
            from its own specification.

    Raises:
        ParameterError: if ``label`` is empty.
    """
    if not label.strip():
        raise ParameterError("an uncertain parameter needs a label to be addressed by")
    return Uncertain(base, label, prior)
