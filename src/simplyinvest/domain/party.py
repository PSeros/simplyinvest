"""The investor, and the facts a domain rule may require of them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, get_type_hints, runtime_checkable

from simplyinvest.errors import PartyFactsMissingError

__all__ = [
    "Anonymous",
    "Party",
    "require",
]


@runtime_checkable
class Party(Protocol):
    """Whoever is putting up the money."""

    @property
    def name(self) -> str:
        """What to call them in a report."""
        ...


@dataclass(frozen=True)
class Anonymous(Party):
    """An investor about whom nothing in particular is known."""

    name: str = "investor"


def _members(protocol: type) -> list[str]:
    """Every attribute a protocol requires."""
    declared = set(getattr(protocol, "__protocol_attrs__", set()))
    if not declared:
        declared = {
            name for klass in protocol.__mro__ for name in vars(klass) if not name.startswith("_")
        } | set(get_type_hints(protocol))
    return sorted(declared)


def require[P: Party](party: Party, facts: type[P]) -> P:
    """Narrow ``party`` to the facts ``facts`` requires.

    Args:
        party: The investor the model was given.
        facts: A protocol declaring what a rule needs to know.

    Returns:
        The same party, typed as carrying those facts.

    Raises:
        PartyFactsMissingError: if the party does not satisfy ``facts``.
    """
    if isinstance(party, facts):
        return party
    missing = [name for name in _members(facts) if not hasattr(party, name)]
    raise PartyFactsMissingError(
        f"this rule needs a party satisfying {facts.__name__}, but {type(party).__name__} "
        f"is missing {missing or 'nothing it can name — check the protocol is runtime_checkable'}"
        f".  Supply a party that carries {', '.join(_members(facts))}."
    )
