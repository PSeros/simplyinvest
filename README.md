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

## Vehicles

`simplyinvest.car` supplies the distance quantity `"km"`, the carriers that
price it, and the German incentives that attach to a purchase. Nothing in the
core knows any of it.

```python
from datetime import date

from simplyinvest import Alternative, GeometricDecline, Term, Timeline, compare
from simplyinvest.car import (
    CarOperating,
    Electricity,
    Household,
    Mileage,
    Petrol,
    Propulsion,
    Vehicle,
)
from simplyinvest.car.incentives import (
    CirculationTaxExemption,
    GhgQuota,
    PurchasePremium,
)
from simplyinvest.financing import CashPurchase

electric = Vehicle(
    name="electric",
    price=42_000,
    propulsion=Propulsion.BEV,
    # Consumption is measured at the battery; the loss is billed at the meter.
    energy=Electricity(
        consumption=17.5, price=0.31, public_price=0.55, home_share=0.8, charging_loss=0.10
    ),
    residual=GeometricDecline(0.15),
    insurance=780,
    maintenance=350,
    circulation_tax=180,
    infrastructure_cost=1_400,
    first_registration=date(2026, 4, 1),
)
petrol = Vehicle(
    name="petrol",
    price=33_500,
    propulsion=Propulsion.ICE,
    energy=Petrol(consumption=6.4, price=1.79, real_world_factor=1.15),
    residual=GeometricDecline(0.13),
    insurance=640,
    maintenance=620,
    circulation_tax=190,
)

timeline = Timeline(
    Term.of_years(8),
    rate=0.03,
    start_date=date(2026, 4, 1),
    escalations={"energy": 0.03, "running_cost": 0.02},
)

result = compare(
    [
        Alternative(
            "electric",
            (
                CashPurchase().bind(electric),
                CarOperating(electric),
                PurchasePremium().bind(electric),
                GhgQuota(annual_amount=300).bind(electric),
                CirculationTaxExemption().bind(electric),
            ),
        ),
        Alternative("petrol", (CashPurchase().bind(petrol), CarOperating(petrol))),
    ],
    timeline,
    party=Household(taxable_income=52_000, children=1),
    usage=Mileage(annual_km=15_000),
)

print(result.verdict())
# > electric is better than petrol by 6,807.98 in present value.

# The energy per kilometre, at period one.
print(f"{petrol.energy.cost_per_km(timeline)[1]:.4f}")
# > 0.1317
print(f"{electric.energy.cost_per_km(timeline)[1]:.4f}")
# > 0.0696

# Equivalent annual cost over the distance actually driven.
print(f"{result.ranking()[0].cost_per_unit('km'):.3f}")
# > 0.557
```

The purchase premium is means-tested, so it reads the buyer's income and
dependants off a published matrix. Ask for it without a buyer who carries those
facts and it names what is missing rather than failing on an attribute deep in a
run; sweep the income and every draw is looked up in its own band.

The circulation-tax exemption is keyed on first registration, not on purchase,
so a used vehicle inherits only the unexpired remainder — counted in whole
months against the scheme's end date, never in average days. It is modelled as a
credit against the tax rather than by zeroing it, so both stay visible in the
breakdown and sum to nothing while the exemption runs.

## Charts and frames

`simplyinvest.report` turns a result object into a pandas frame or a matplotlib
axes. It composes no prose: the frames are tidy and the charts are plain, so a
notebook or a slide decides the presentation.

```python
from simplyinvest import GeometricDecline, Term, Timeline
from simplyinvest.appraisal import Alternative, Case
from simplyinvest.car import CarOperating, Electricity, Mileage, Petrol, Propulsion, Vehicle
from simplyinvest.financing import CashPurchase
from simplyinvest.report import breakdown_chart, breakdown_frame, ranking_frame

electric = Vehicle(
    name="electric",
    price=42_000.0,
    propulsion=Propulsion.BEV,
    energy=Electricity(consumption=17.5, price=0.31),
    residual=GeometricDecline(0.15),
)
petrol = Vehicle(
    name="petrol",
    price=33_500.0,
    propulsion=Propulsion.ICE,
    energy=Petrol(consumption=6.4, price=1.79),
    residual=GeometricDecline(0.13),
)
timeline = Timeline(Term.of_years(8), rate=0.03, escalations={"energy": 0.03})

result = Case(
    alternatives=tuple(
        Alternative(car.name, sources=(CashPurchase().bind(car), CarOperating(car)))
        for car in (electric, petrol)
    ),
    timeline=timeline,
    usage=Mileage(annual_km=15_000.0),
).run()

# A column per alternative, a row per component, summing to the present value.
parts = breakdown_frame(result)
print(list(parts.index))
# > ['capital/purchase', 'operating/energy', 'terminal/residual']
print(f"{parts['electric'].sum():,.2f}")
# > -39,388.16

# Best first, with every measure the appraisal carries.
print(ranking_frame(result).index[0])
# > petrol

# A chart draws on the axes it is given, or on one of its own, and returns it.
print(breakdown_chart(result).get_xlabel())
# > present value
```

| Chart | What it draws |
| --- | --- |
| `breakdown_chart` | Present value per component, grouped by alternative |
| `cumulative_chart` | The running discounted total, with the period the ranking turns |
| `schedule_chart` | Each instalment split into interest and principal |
| `balance_chart` | What is still owed after each instalment |
| `distribution_chart` | Every alternative's spread of simulated outcomes |
| `differential_chart` | Trial-by-trial margin, coloured by which side won |
| `tornado_chart` | Each parameter's swing around the base case |
| `sweep_chart` | Present value against one parameter, with its switch point |
| `scenario_chart` | Present value in every named scenario |

Each has a frame beside it — `flows_frame`, `breakdown_frame`, `ranking_frame`,
`schedule_frame`, `simulation_frame`, `draws_frame`, `sweep_frame`,
`tornado_frame`, `scenario_frame`. `flows_frame` is the long one: a row for every
flow and period in which money actually moves, with its role, its description
and its discounted value.

Install what you use: `uv add "simplyinvest[frames]"` for the frames,
`"simplyinvest[viz]"` for the charts. Importing `simplyinvest` pulls in neither.

## Worked examples

Three notebooks under `examples/`, runnable with `uv sync --extra examples`:

| Notebook | What it shows |
| --- | --- |
| `loan_vs_cash.ipynb` | Paying outright against borrowing: component breakdown, cumulative discounted cash flow, and the amortisation schedule split into interest and principal |
| `uncertain_machine.ipynb` | The same appraisal under uncertainty: overlapping outcome distributions, the differential stream, a tornado, a switch point and named scenarios |
| `electric_vs_petrol.ipynb` | A whole domain end to end: what a kilometre costs on each carrier, the three public benefits as a waterfall to the effective price, the year the electric car pulls ahead, the mileage that decides it, and the cliff the means test puts in the answer |

Every cell is executed by the test suite, so a figure in a notebook cannot drift
away from the code that produced it.

## Status

Early development. The core is built: timeline and durations, cash flows,
metrics, the domain-extension contract, financing with a full amortisation
schedule, tax and incentive contracts, and comparison with replacement chains.

The uncertainty layer is built too: distributions, parameter addressing,
correlated sampling, simulation, one-way sweeps, tornados, switch points and
scenarios.

The `car` domain is built: vehicles, energy carriers, distance-based running
costs, a distance-settled lease, and the purchase premium, quota credit and
circulation-tax exemption.

Charts and frames are built, over every result object the package returns.

Next: the `pv` domain.

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
