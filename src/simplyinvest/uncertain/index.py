"""Every uncertain parameter in a model, addressed once."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from simplyinvest.errors import ParameterError

from .lens import Attribute, Item, Lens, Step, sweepable
from .mark import Uncertain

if TYPE_CHECKING:
    from simplyinvest.money import Quantity

    from .distribution import Distribution

__all__ = ["ParameterIndex"]

MAX_DEPTH = 24
"""How deep a model tree is walked before the search gives up."""


@dataclass(frozen=True)
class ParameterIndex(Mapping[str, Lens]):
    """The addressable parameters of one model, with any priors they declared.

    Args:
        lenses: Each parameter's address, by label.
        priors: The distribution a parameter was declared with, by label.
    """

    lenses: Mapping[str, Lens] = field(default_factory=dict)
    priors: Mapping[str, Distribution] = field(default_factory=dict)

    @classmethod
    def of(cls, root: Any) -> ParameterIndex:
        """Every :func:`~simplyinvest.uncertain.mark.uncertain` value inside ``root``.

        Raises:
            ParameterError: if two parameters share a label.
        """
        found: list[tuple[tuple[Step, ...], Uncertain]] = []
        _walk(root, (), 0, found)

        lenses: dict[str, Lens] = {}
        priors: dict[str, Distribution] = {}
        declared: dict[str, float] = {}
        for steps, mark in found:
            label = mark.label
            addressed = Lens.at(steps, label, sweepable(mark, label))
            if label in lenses:
                _check_same_parameter(label, declared[label], float(mark))
                addressed = lenses[label].merged(addressed)
            lenses[label] = addressed
            declared[label] = float(mark)
            if mark.prior is not None:
                priors[label] = mark.prior
        return cls(lenses=lenses, priors=priors)

    def __getitem__(self, label: str) -> Lens:
        try:
            return self.lenses[label]
        except KeyError:
            raise ParameterError(
                f"no parameter is labelled {label!r}; this model offers {self.labels}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self.lenses)

    def __len__(self) -> int:
        return len(self.lenses)

    @property
    def labels(self) -> tuple[str, ...]:
        """Every parameter's label, in a stable order."""
        return tuple(sorted(self.lenses))

    def with_lens(self, addressed: Lens, prior: Distribution | None = None) -> ParameterIndex:
        """This index plus one parameter addressed by :func:`~.lens.lens`."""
        priors = dict(self.priors)
        if prior is not None:
            priors[addressed.label] = prior
        return ParameterIndex({**self.lenses, addressed.label: addressed}, priors)

    def distributions(
        self, spec: Mapping[str, Distribution] | None = None
    ) -> dict[str, Distribution]:
        """Each parameter's distribution, with ``spec`` overriding any prior.

        Raises:
            ParameterError: if ``spec`` names an unknown parameter, or any
                parameter is left without a distribution.
        """
        given = dict(spec or {})
        unknown = sorted(set(given) - set(self.lenses))
        if unknown:
            raise ParameterError(
                f"the specification names parameters this model does not have: {unknown}.  "
                f"It offers {self.labels}."
            )
        resolved = {**self.priors, **given}
        missing = sorted(set(self.lenses) - set(resolved))
        if missing:
            raise ParameterError(
                f"these parameters have no distribution: {missing}.  Give each one a prior "
                f"where it is declared, or a distribution in the simulation's specification."
            )
        return resolved

    def apply(self, root: Any, values: Mapping[str, Quantity]) -> Any:
        """``root`` with every named parameter replaced by its value."""
        rebuilt = root
        for label, value in values.items():
            rebuilt = self[label].set(rebuilt, value)
        return rebuilt


def _check_same_parameter(label: str, declared: float, found: float) -> None:
    """Confirm two marks sharing a label describe one quantity.

    Raises:
        ParameterError: if they were declared at different values.
    """
    if declared != found:
        raise ParameterError(
            f"two different numbers are both labelled {label!r} — {declared!r} and {found!r} — "
            f"so a draw could not say which it meant.  Rename one of them, or give both the "
            f"same declared value if they really are the same quantity."
        )


def _walk(
    node: Any, steps: tuple[Step, ...], depth: int, found: list[tuple[tuple[Step, ...], Uncertain]]
) -> None:
    """Collect every marked value under ``node``, recording how it is reached."""
    if isinstance(node, Uncertain):
        found.append((steps, node))
        return
    if depth >= MAX_DEPTH:
        return
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        for member in dataclasses.fields(node):
            _walk(getattr(node, member.name), (*steps, Attribute(member.name)), depth + 1, found)
    elif type(node) in (tuple, list):
        for position, item in enumerate(node):
            _walk(item, (*steps, Item(position)), depth + 1, found)
    elif isinstance(node, Mapping):
        for key, item in node.items():
            _walk(item, (*steps, Item(key)), depth + 1, found)
