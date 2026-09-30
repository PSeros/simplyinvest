# simplyinvest

Domain-agnostic investment appraisal: NPV, IRR, financing, tax and uncertainty,
with pluggable domains.

An investment is a stream of timed cash flows. Everything here follows from
that — discount the stream, rank alternatives on the result, and let each domain
say *how* its cash flows arise without the engine ever learning what a car or a
solar panel is.

```console
uv add simplyinvest              # generic investments
uv add "simplyinvest[car]"       # vehicles
uv add "simplyinvest[pv]"        # photovoltaics
```

## The smallest complete example

```python
from simplyinvest import Amount, Term, Timeline
from simplyinvest.cashflow import CashFlowSeries, Frequency, OneOff, Recurring

timeline = Timeline(Term.of_years(3), periods_per_year=1, rate=0.10)

project = CashFlowSeries.of(
    OneOff(Amount.paid(1_000)),  # 1,000 out today
    Recurring(Amount.received(500), Frequency.ANNUAL),  # 500 back each year
)

print(f"{project.pv(timeline):,.2f}")
# > 243.43
```

Money is an `Amount` — a magnitude and a direction — so a cost cannot silently
become a benefit. `Amount.paid(-500)` raises rather than quietly reversing.

## Comparing alternatives

```python
from dataclasses import dataclass

from simplyinvest import Amount, Term, Timeline, compare
from simplyinvest.appraisal import Alternative
from simplyinvest.domain import GeometricDecline
from simplyinvest.financing import AnnuityLoan, CashPurchase


@dataclass(frozen=True)
class Machine:
    """A capital asset."""

    name: str
    price: float
    age_at_acquisition: Term = Term.ZERO
    economic_life: Term | None = None

    @property
    def capital_cost(self) -> Amount:
        return Amount.paid(self.price)

    @property
    def setup_cost(self) -> Amount:
        return Amount.zero()

    def residual_value(self, *, held: Term) -> Amount:
        return Amount.received(
            GeometricDecline(0.18).value_after(
                self.price, held=held, age_at_acquisition=self.age_at_acquisition
            )
        )


mill = Machine("CNC mill", price=86_000.0)
timeline = Timeline(Term.of_years(8), rate=0.05)

result = compare(
    [
        Alternative("Pay cash", sources=(CashPurchase().bind(mill),)),
        Alternative(
            "Borrow at 6.5%",
            sources=(AnnuityLoan(rate=0.065, term=Term.of_years(5)).bind(mill),),
        ),
    ],
    timeline,
)

print(result.best().name)
# > Pay cash
print(result.verdict())
# > Pay cash is better than Borrow at 6.5% by 3,010.22 in present value.
```

`result.to_markdown()` gives the ranked table, the verdict, and every obligation
the figures do not price — here, that borrowing above the discount rate destroys
value.

## What the design insists on

**A duration is a `Term`, never a number of months.** Only a `Timeline` turns one
into periods, and it refuses spans that do not land on the grid:

```python
from simplyinvest import Term, Timeline

quarterly = Timeline(Term.of_years(5), periods_per_year=4)
print(quarterly.periods_in(Term.of_months(36)))
# > 12
```

A five-month term on that grid raises `TermNotRepresentableError` rather than
rounding to something plausible.

**Real and nominal never mix.** A real discount rate must state the inflation it
is real to; a nominal rate applied to flows that never grow is refused outright,
because that combination understates every later flow and looks fine doing it.

**The core holds no jurisdiction.** No VAT rate, no subsidy table, no statutory
date — those belong to whichever domain claims them, under `<domain>/incentives/`.
An architecture test enforces it.

## Uncertainty

A number that is not really known is marked where it is written, and keeps
working as a number everywhere else:

```python
from dataclasses import dataclass

from simplyinvest import (
    Alternative,
    Amount,
    Case,
    CashFlowSeries,
    Component,
    Context,
    Recurring,
    Role,
    Term,
    Timeline,
)
from simplyinvest.financing import CashPurchase
from simplyinvest.money import Quantity
from simplyinvest.uncertain import (
    LogNormal,
    Normal,
    lens,
    ref,
    simulate,
    switch_point,
    tornado,
    uncertain,
)


@dataclass(frozen=True)
class Machine:
    """A capital asset, worth nothing at the end."""

    name: str
    price: Quantity
    age_at_acquisition: Term = Term.ZERO
    economic_life: Term | None = None

    @property
    def capital_cost(self) -> Amount:
        return Amount.paid(self.price)

    @property
    def setup_cost(self) -> Amount:
        return Amount.zero()

    def residual_value(self, *, held: Term) -> Amount:
        return Amount.zero()


@dataclass(frozen=True)
class Hire:
    """A monthly rate for the use of a machine."""

    rate: Quantity = 0.0

    def flows(self, ctx: Context) -> CashFlowSeries:
        return CashFlowSeries.of(
            Recurring(Amount.paid(self.rate), label=Component(Role.OPERATING, "hire"))
        )

    def constraints(self, ctx: Context) -> tuple[str, ...]:
        return ()

    @property
    def supports_batch(self) -> bool:
        return True


mill = Machine(
    "CNC mill", price=uncertain(86_000.0, "price", LogNormal.from_mean_cv(86_000.0, 0.09))
)
case = Case(
    alternatives=(
        Alternative("Buy", sources=(CashPurchase().bind(mill),)),
        Alternative("Hire", sources=(Hire(uncertain(1_150.0, "rate", Normal(1_150.0, 90.0))),)),
    ),
    timeline=Timeline(Term.of_years(8), rate=0.05),
)

run = simulate(case, n=10_000, seed=20260930)
print(
    f"Buy wins {run.win_share()['Buy']:.0%} of trials, and is worth {run.expected_npv('Buy'):,.2f} on average"
)
# > Buy wins 69% of trials, and is worth -86,101.40 on average

# What the answer turns on, and where the ranking actually changes.
print(tornado(case, on="Buy").widest().label)
# > price
print(switch_point(case, "rate", better="Hire", worse="Buy", bracket=(200.0, 4_000.0)))
# > Hire leads while rate is below 1,084.21, Buy above it

# Anything not marked is reached with a checked path.
moved = lens(ref(case).timeline.rate, "discount_rate").set(case, 0.09)
print(f"{moved.timeline.rate} vs {case.timeline.rate}")
# > 0.09 vs 0.05
```

`simulate` finds every mark itself. Where a parameter appears more than once —
an asset's price is copied into the financing plan bound to it — all of its
addresses move together, so a draw cannot leave half the model behind.

Draws are correlated through a Gaussian copula, so imposing a correlation never
quietly reshapes a marginal. Where a source builds a different stream for every
draw, as an amortisation schedule does, it says so and the trials are resolved
one at a time instead.

A `switch_point` names which alternative leads on each side rather than handing
back a bare number to be read the wrong way round. A mistyped `ref` path fails
where it was typed, naming the fields that do exist, rather than inside trial 417.

## Worked examples

Two notebooks under `examples/`, runnable with `uv sync --extra examples`:

| Notebook | What it shows |
| --- | --- |
| `loan_vs_cash.ipynb` | Paying outright against borrowing: component breakdown, cumulative discounted cash flow, and the amortisation schedule split into interest and principal |
| `uncertain_machine.ipynb` | The same appraisal under uncertainty: overlapping outcome distributions, the differential stream, a tornado, a switch point and named scenarios |

Every cell is executed by the test suite, so a figure in a notebook cannot drift
away from the code that produced it.

## Status

Early development. The core is built: timeline and durations, cash flows,
metrics, the domain-extension contract, financing with a full amortisation
schedule, tax and incentive contracts, and comparison with replacement chains.

The uncertainty layer is built too: distributions, parameter addressing,
correlated sampling, simulation, one-way sweeps, tornados, switch points and
scenarios.

Next: charts and frames, then the `car` and `pv` domains.

## Development

```console
uv sync --extra examples
uv run pytest
uv run ruff check . && uv run mypy
uv run jupyter lab examples/
```

`uv sync` alone is enough for the library and its tests; the `examples` extra
adds JupyterLab, matplotlib and pandas. The notebook tests skip without it.

## License

MIT
