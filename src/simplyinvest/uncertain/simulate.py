"""Running a case many times over, with its uncertain parameters drawn."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from simplyinvest.errors import ParameterError
from simplyinvest.money import format_money as _money

from .index import ParameterIndex
from .sampler import uniforms

if TYPE_CHECKING:
    import numpy.typing as npt

    from simplyinvest.appraisal import Case

    from .distribution import Distribution
    from .sampler import Correlations

__all__ = ["Mode", "Simulation", "simulate"]

type Mode = Literal["auto", "batch", "loop"]

DEFAULT_TRIALS = 5_000
"""How many trials a simulation runs unless another count is given."""

DEFAULT_PERCENTILES = (5.0, 25.0, 50.0, 75.0, 95.0)
"""The percentiles a simulation reports unless others are asked for."""

DEFAULT_RISK_LEVEL = 0.05
"""The tail a value at risk is quoted at unless another is given."""


def simulate(
    case: Case,
    *,
    n: int = DEFAULT_TRIALS,
    spec: Mapping[str, Distribution] | None = None,
    correlation: Correlations | None = None,
    seed: int | None = None,
    mode: Mode = "auto",
) -> Simulation:
    """Run ``case`` ``n`` times with its uncertain parameters drawn.

    Args:
        case: The alternatives to weigh, carrying marked parameters.
        n: How many trials to run.
        spec: Distributions by label, overriding any declared prior.
        correlation: Correlation coefficients by pair of labels.
        seed: Fixes the draws, so a run repeats exactly.
        mode: ``"batch"`` resolves every trial in one pass and refuses when a
            source cannot, ``"loop"`` resolves them one at a time, ``"auto"``
            batches where every alternative allows it.

    Raises:
        ParameterError: if nothing is marked uncertain, or a parameter has no
            distribution.
        ValueError: if ``mode`` is ``"batch"`` and an alternative refuses it.
    """
    index = ParameterIndex.of(case)
    if not index:
        raise ParameterError(
            "nothing in this case is marked uncertain, so every trial would give the same "
            "answer.  Wrap a number in uncertain(value, label, prior), or add a lens."
        )
    distributions = index.distributions(spec)
    labels = index.labels
    rng = np.random.default_rng(seed)
    draws = {
        label: distributions[label].ppf(drawn)
        for label, drawn in uniforms(rng, n, labels, correlation).items()
    }

    names = tuple(alternative.name for alternative in case.alternatives)
    batched = _batches(case, mode)
    npv = _in_one_pass(case, index, draws, n) if batched else _trial_by_trial(case, index, draws, n)
    return Simulation(names=names, npv=npv, draws=draws, batched=batched)


def _batches(case: Case, mode: Mode) -> bool:
    """Whether ``case`` may be resolved for every trial in one pass."""
    if mode == "loop":
        return False
    refusing = tuple(
        alternative.name
        for alternative in case.alternatives
        if not getattr(alternative, "supports_batch", False)
    )
    if mode == "batch" and refusing:
        raise ValueError(
            f"these alternatives build a different stream for every draw, so they cannot be "
            f"resolved in one pass: {list(refusing)}.  Run with mode='loop', or mode='auto' "
            f"to let each case choose."
        )
    return not refusing


def _in_one_pass(
    case: Case, index: ParameterIndex, draws: Mapping[str, npt.NDArray[np.float64]], n: int
) -> npt.NDArray[np.float64]:
    """Every trial resolved at once, with the draws carried as arrays."""
    result = index.apply(case, draws).run()
    return np.stack(
        [
            np.broadcast_to(np.asarray(result[name].npv, dtype=np.float64), (n,))
            for name in result.names
        ]
    )


def _trial_by_trial(
    case: Case, index: ParameterIndex, draws: Mapping[str, npt.NDArray[np.float64]], n: int
) -> npt.NDArray[np.float64]:
    """Each trial resolved on its own, for cases whose structure moves with the draw."""
    out = np.empty((len(case.alternatives), n), dtype=np.float64)
    for trial in range(n):
        values: dict[str, Any] = {label: float(drawn[trial]) for label, drawn in draws.items()}
        result = index.apply(case, values).run()
        out[:, trial] = [float(result[name].npv) for name in result.names]
    return out


@dataclass(frozen=True)
class Simulation:
    """What a case came to over many trials.

    Args:
        names: The alternatives, in the order the case gave them.
        npv: Net present value per alternative and trial, shape
            ``(alternatives, trials)``.
        draws: The value each parameter took in each trial.
        batched: Whether every trial was resolved in one pass.
    """

    names: tuple[str, ...]
    npv: npt.NDArray[np.float64]
    draws: Mapping[str, npt.NDArray[np.float64]]
    batched: bool

    @property
    def trials(self) -> int:
        """How many trials were run."""
        return int(self.npv.shape[1])

    def values(self, name: str) -> npt.NDArray[np.float64]:
        """Every trial's net present value for one alternative.

        Raises:
            KeyError: if no alternative goes by that name.
        """
        if name not in self.names:
            raise KeyError(f"no alternative named {name!r}; this simulation ran {self.names}")
        return np.asarray(self.npv[self.names.index(name)], dtype=np.float64)

    # ------------------------------------------------------- one alternative

    def expected_npv(self, name: str) -> float:
        """Mean net present value across the trials."""
        return float(self.values(name).mean())

    def npv_percentiles(
        self, name: str, levels: Sequence[float] = DEFAULT_PERCENTILES
    ) -> dict[float, float]:
        """Net present value at each percentile."""
        found = np.percentile(self.values(name), levels)
        return {float(level): float(value) for level, value in zip(levels, found, strict=True)}

    def probability_npv_positive(self, name: str) -> float:
        """Share of trials in which this alternative is worth more than nothing."""
        return float((self.values(name) > 0.0).mean())

    def value_at_risk(self, name: str, *, level: float = DEFAULT_RISK_LEVEL) -> float:
        """The net present value the worst ``level`` of trials fall below."""
        return float(np.percentile(self.values(name), 100.0 * level))

    # ------------------------------------------------------ two alternatives

    def probability(self, better: str, worse: str) -> float:
        """Share of trials in which ``better`` beats ``worse``."""
        return float((self.values(better) > self.values(worse)).mean())

    def standard_error(self, better: str, worse: str) -> float:
        """How far :meth:`probability` could be out, for this many trials."""
        share = self.probability(better, worse)
        return float(np.sqrt(share * (1.0 - share) / self.trials))

    # ------------------------------------------------------------ the ranking

    def win_share(self) -> dict[str, float]:
        """Share of trials each alternative comes out on top."""
        winners = np.argmax(self.npv, axis=0)
        return {name: float((winners == index).mean()) for index, name in enumerate(self.names)}

    def regret(self, name: str) -> float:
        """Mean value given up by choosing this alternative and being wrong."""
        return float((self.npv.max(axis=0) - self.values(name)).mean())

    def best_by_expected_value(self) -> str:
        """The alternative with the highest mean net present value."""
        return max(self.names, key=self.expected_npv)

    def best_by_regret(self) -> str:
        """The alternative that gives up least when it turns out to be wrong."""
        return min(self.names, key=self.regret)

    def rules_agree(self) -> bool:
        """Whether expected value and regret pick the same alternative."""
        return self.best_by_expected_value() == self.best_by_regret()

    # ----------------------------------------------------------------- report

    def verdict(self) -> str:
        """One sentence stating what the trials came to."""
        winner = self.best_by_expected_value()
        others = [name for name in self.names if name != winner]
        runner_up = max(others, key=self.expected_npv)
        share = self.probability(winner, runner_up)
        error = self.standard_error(winner, runner_up)
        agreement = (
            ""
            if self.rules_agree()
            else f"  Least regret points at {self.best_by_regret()} instead, so the choice "
            f"turns on how much a bad outcome matters."
        )
        return (
            f"{winner} has the higher expected value, and beats {runner_up} in "
            f"{share:.1%} of {self.trials:,} trials (±{error:.1%})." + agreement
        )

    def to_markdown(self) -> str:
        """A table of each alternative's distribution, and the verdict."""
        lines = [
            "| Alternative | Expected | P5 | Median | P95 | Wins | Regret |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        shares = self.win_share()
        for name in sorted(self.names, key=self.expected_npv, reverse=True):
            spread = self.npv_percentiles(name, (5.0, 50.0, 95.0))
            lines.append(
                f"| {name} | {_money(self.expected_npv(name))} | {_money(spread[5.0])} "
                f"| {_money(spread[50.0])} | {_money(spread[95.0])} | {shares[name]:.1%} "
                f"| {_money(self.regret(name))} |"
            )
        lines += ["", self.verdict()]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.verdict()
