"""Addressing one number inside a frozen model tree."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from simplyinvest.errors import NotSweepableError, ParameterError

__all__ = ["Attribute", "Item", "Lens", "Step", "lens", "ref"]

_PROXY_FIELDS = ("_lens_target", "_lens_steps")


@dataclass(frozen=True, slots=True)
class Attribute:
    """A step onto a named field."""

    name: str

    def __str__(self) -> str:
        return f".{self.name}"


@dataclass(frozen=True, slots=True)
class Item:
    """A step onto a position or a key."""

    key: Any

    def __str__(self) -> str:
        return f"[{self.key!r}]"


type Step = Attribute | Item


def describe(steps: tuple[Step, ...]) -> str:
    """The path ``steps`` walk, written the way it was typed."""
    return "".join(str(step) for step in steps).lstrip(".")


@dataclass(frozen=True, slots=True)
class Lens:
    """Where one parameter lives in a frozen tree.

    A parameter may appear at several addresses at once, as an asset's price
    does once a financing plan has been bound to it.  All of them move together.

    Args:
        paths: Every path from the root to the parameter.
        label: What to call it.
        leaf_type: The type found there when the tree was walked.
    """

    paths: tuple[tuple[Step, ...], ...]
    label: str
    leaf_type: type

    @classmethod
    def at(cls, steps: tuple[Step, ...], label: str, leaf_type: type) -> Lens:
        """A lens on the single path ``steps``."""
        return cls(paths=(steps,), label=label, leaf_type=leaf_type)

    def merged(self, other: Lens) -> Lens:
        """This lens also addressing ``other``'s paths."""
        return Lens(
            paths=self.paths + tuple(p for p in other.paths if p not in self.paths),
            label=self.label,
            leaf_type=self.leaf_type,
        )

    def get(self, root: Any) -> Any:
        """The value this addresses in ``root``.

        Raises:
            ParameterError: if the path does not resolve.
        """
        node = root
        steps = self.paths[0]
        for depth, step in enumerate(steps):
            try:
                node = getattr(node, step.name) if isinstance(step, Attribute) else node[step.key]
            except (AttributeError, KeyError, IndexError, TypeError) as missing:
                raise ParameterError(
                    f"{self.label!r} addresses {describe(steps)}, but "
                    f"{describe(steps[: depth + 1])} does not resolve in this tree"
                ) from missing
        return node

    def set(self, root: Any, value: Any) -> Any:
        """``root`` with every address holding ``value``, rebuilt along those paths only."""
        rebuilt = root
        for steps in self.paths:
            rebuilt = _replace_at(rebuilt, steps, value, self.label)
        return rebuilt

    @property
    def path(self) -> str:
        """Where this parameter lives, written the way it was typed."""
        return " and ".join(describe(steps) for steps in self.paths)

    def __str__(self) -> str:
        return f"{self.label} at {self.path}"


def _replace_at(node: Any, steps: tuple[Step, ...], value: Any, label: str) -> Any:
    """``node`` with the value at ``steps`` swapped, rebuilding only that spine."""
    if not steps:
        return value
    head, rest = steps[0], steps[1:]
    if isinstance(head, Attribute):
        child = _replace_at(getattr(node, head.name), rest, value, label)
        return _with_attribute(node, head.name, child, label)
    child = _replace_at(node[head.key], rest, value, label)
    return _with_item(node, head.key, child, label)


def _with_attribute(node: Any, name: str, value: Any, label: str) -> Any:
    """``node`` with field ``name`` replaced."""
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        return dataclasses.replace(node, **{name: value})
    raise ParameterError(
        f"{label!r} passes through {type(node).__name__}, which is not a dataclass, "
        f"so {name!r} cannot be replaced.  Make it a frozen dataclass, or address a "
        f"parameter that sits inside one."
    )


def _with_item(node: Any, key: Any, value: Any, label: str) -> Any:
    """``node`` with the entry at ``key`` replaced."""
    if type(node) is tuple:
        items = list(node)
        items[key] = value
        return tuple(items)
    if type(node) is list:
        return [*node[:key], value, *node[key + 1 :]]
    if type(node) is dict:
        return {**node, key: value}
    raise ParameterError(
        f"{label!r} passes through {type(node).__name__}, which this package cannot "
        f"rebuild.  Use a tuple, list or dict, or address a parameter that sits inside one."
    )


def sweepable(value: Any, label: str) -> type:
    """The type of ``value``, once it is known to be able to hold a draw.

    Raises:
        NotSweepableError: if ``value`` is not a number or an array of them.
    """
    if isinstance(value, bool) or not isinstance(value, int | float | np.floating | np.ndarray):
        raise NotSweepableError(
            f"{label!r} addresses a {type(value).__name__}, which cannot hold a draw.  "
            f"A swept parameter is a number."
        )
    return type(value)


class _Ref:
    """A stand-in for a model tree that records how it is reached."""

    def __init__(self, target: Any, steps: tuple[Step, ...]) -> None:
        self._lens_target = target
        self._lens_steps = steps

    def __getattr__(self, name: str) -> _Ref:
        if name in _PROXY_FIELDS:
            raise AttributeError(name)
        target = object.__getattribute__(self, "_lens_target")
        try:
            child = getattr(target, name)
        except AttributeError as missing:
            raise ParameterError(
                f"{type(target).__name__} has no field {name!r}; it offers {_fields_of(target)}"
            ) from missing
        return _Ref(child, (*self._lens_steps, Attribute(name)))

    def __getitem__(self, key: Any) -> _Ref:
        try:
            child = self._lens_target[key]
        except (KeyError, IndexError, TypeError) as missing:
            raise ParameterError(
                f"{type(self._lens_target).__name__} has no entry {key!r}"
            ) from missing
        return _Ref(child, (*self._lens_steps, Item(key)))

    def __repr__(self) -> str:
        return f"ref({describe(self._lens_steps) or type(self._lens_target).__name__})"


def _fields_of(target: Any) -> str:
    """The names ``target`` does offer, for an error message."""
    if dataclasses.is_dataclass(target) and not isinstance(target, type):
        return ", ".join(field.name for field in dataclasses.fields(target))
    if isinstance(target, Mapping):
        return ", ".join(repr(key) for key in target)
    return ", ".join(name for name in dir(target) if not name.startswith("_"))


def ref(root: Any) -> Any:
    """A proxy over ``root`` that records the path taken through it.

    Every step is checked against the live tree as it is taken, so a mistyped
    field fails where it was typed.
    """
    return _Ref(root, ())


def lens(path: Any, label: str | None = None) -> Lens:
    """The address a :func:`ref` path walked.

    Args:
        path: An attribute or item path taken through a :func:`ref` proxy.
        label: What to call the parameter.  Defaults to the path itself.

    Raises:
        ParameterError: if ``path`` is not a :func:`ref` path or is empty.
        NotSweepableError: if it does not end on a number.
    """
    if not isinstance(path, _Ref):
        raise ParameterError(
            f"lens() takes a path through ref(...), got {type(path).__name__}.  "
            f"Write lens(ref(case).alternatives[0].tax.rate)."
        )
    steps = path._lens_steps
    if not steps:
        raise ParameterError("lens() needs a path into the tree, not the root itself")
    name = label or describe(steps)
    return Lens.at(steps, name, sweepable(path._lens_target, name))
