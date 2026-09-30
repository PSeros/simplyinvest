"""Named annual growth rates."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .conventions import fisher_nominal

__all__ = ["EscalationSet"]


@dataclass(frozen=True, slots=True)
class EscalationSet(Mapping[str, float]):
    """Annual growth rates by name, where an undeclared name reads as ``0.0``.

    Args:
        rates: Growth rate per name.

    Raises:
        ValueError: if a rate is a fall of 100% or more per year.
    """

    rates: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cleaned = {str(key): float(value) for key, value in dict(self.rates).items()}
        for key, value in cleaned.items():
            if value <= -1.0:
                raise ValueError(
                    f"escalation {key!r} is {value}, which is a fall of 100% or more per year"
                )
        object.__setattr__(self, "rates", MappingProxyType(cleaned))

    @classmethod
    def of(cls, **rates: float) -> EscalationSet:
        """Build from keyword arguments."""
        return cls(rates)

    def __getitem__(self, key: str) -> float:
        """The rate declared for ``key``, or ``0.0`` if none was."""
        return self.rates.get(key, 0.0)

    def __contains__(self, key: object) -> bool:
        """Whether ``key`` was declared."""
        return key in self.rates

    def __iter__(self) -> Iterator[str]:
        return iter(self.rates)

    def __len__(self) -> int:
        return len(self.rates)

    @property
    def any_nonzero(self) -> bool:
        """Whether any declared rate is non-zero."""
        return any(rate != 0.0 for rate in self.rates.values())

    def to_nominal(self, inflation: float) -> EscalationSet:
        """Every rate converted from real to nominal growth."""
        return EscalationSet({k: fisher_nominal(v, inflation) for k, v in self.rates.items()})

    def with_rate(self, key: str, rate: float) -> EscalationSet:
        """A copy carrying one more rate."""
        return EscalationSet({**self.rates, key: rate})

    def __str__(self) -> str:
        if not self.rates:
            return "no escalation"
        return ", ".join(f"{k} {v:+.2%}" for k, v in sorted(self.rates.items()))
