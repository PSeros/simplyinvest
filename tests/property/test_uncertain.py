"""Invariants the uncertainty layer must hold for any inputs."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from simplyinvest.uncertain import (
    Constant,
    Distribution,
    Empirical,
    LogNormal,
    Normal,
    Triangular,
    Uniform,
    correlation_matrix,
    uniforms,
)
from simplyinvest.uncertain.gaussian import norm_cdf, norm_ppf

probabilities = st.floats(min_value=1e-9, max_value=1.0 - 1e-9)

positive = st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False)

spreads = st.floats(min_value=0.01, max_value=0.9)


def shapes() -> st.SearchStrategy[Distribution]:
    """Any distribution this package offers, built from valid parameters."""
    return st.one_of(
        st.builds(Constant, st.floats(min_value=-1e6, max_value=1e6)),
        st.builds(
            lambda low, width: Uniform(low, low + width),
            st.floats(min_value=-1e6, max_value=1e6),
            positive,
        ),
        st.builds(
            lambda low, first, second: Triangular(
                low, low + min(first, second), low + first + second
            ),
            st.floats(min_value=-1e6, max_value=1e6),
            positive,
            positive,
        ),
        st.builds(Normal, st.floats(min_value=-1e6, max_value=1e6), positive),
        st.builds(LogNormal.from_mean_cv, positive, spreads),
        st.builds(
            lambda sample: Empirical(np.asarray(sample, dtype=np.float64)),
            st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1, max_size=20),
        ),
    )


class TestEveryDistribution:
    @given(shape=shapes(), first=probabilities, second=probabilities)
    def test_its_quantiles_never_run_backwards(
        self, shape: Distribution, first: float, second: float
    ) -> None:
        """An inverse CDF is non-decreasing, whatever shape it has."""
        low, high = sorted((first, second))
        assert shape.quantile(low) <= shape.quantile(high) + 1e-9

    @given(shape=shapes(), p=probabilities)
    def test_a_quantile_lands_inside_the_interval_it_bounds(
        self, shape: Distribution, p: float
    ) -> None:
        low, high = shape.interval(low=min(p, 0.5), high=max(p, 0.5))
        assert low <= high + 1e-9

    @given(
        shape=shapes(), n=st.integers(min_value=1, max_value=200), seed=st.integers(0, 2**32 - 1)
    )
    def test_a_sample_has_the_size_asked_for(self, shape: Distribution, n: int, seed: int) -> None:
        drawn = shape.sample(np.random.default_rng(seed), n)
        assert drawn.shape == (n,)
        assert np.all(np.isfinite(drawn))


class TestTheStandardNormal:
    @given(p=probabilities)
    def test_the_quantile_function_inverts_the_distribution_function(self, p: float) -> None:
        assert float(norm_cdf(norm_ppf(np.asarray(p)))) == pytest.approx(p, rel=1e-11, abs=1e-15)

    @given(x=st.floats(min_value=-8.0, max_value=5.0))
    def test_the_distribution_function_inverts_the_quantile_function(self, x: float) -> None:
        """Above about five, float64 no longer resolves the upper tail it would return."""
        assert float(norm_ppf(norm_cdf(np.asarray(x)))) == pytest.approx(x, abs=1e-9)

    @given(x=st.floats(min_value=-8.0, max_value=8.0))
    def test_it_is_symmetric_about_zero(self, x: float) -> None:
        assert float(norm_cdf(np.asarray(x))) + float(norm_cdf(np.asarray(-x))) == pytest.approx(
            1.0
        )


class TestLogNormal:
    @given(mean=positive, cv=spreads)
    def test_it_recovers_the_mean_it_was_built_from(self, mean: float, cv: float) -> None:
        assert LogNormal.from_mean_cv(mean, cv).expectation == pytest.approx(mean, rel=1e-9)

    @given(mean=positive, cv=spreads)
    def test_its_mean_is_above_its_median(self, mean: float, cv: float) -> None:
        """A right skew is the whole reason to reach for this distribution."""
        shape = LogNormal.from_mean_cv(mean, cv)
        assert shape.expectation > shape.median

    @given(low=positive, factor=st.floats(min_value=1.1, max_value=20.0))
    def test_it_hits_the_quantiles_it_was_elicited_from(self, low: float, factor: float) -> None:
        shape = LogNormal.from_quantiles(low=low, high=low * factor)
        assert shape.quantile(0.1) == pytest.approx(low, rel=1e-9)
        assert shape.quantile(0.9) == pytest.approx(low * factor, rel=1e-9)

    @given(p=probabilities, mean=positive, cv=spreads)
    def test_every_draw_is_positive(self, p: float, mean: float, cv: float) -> None:
        assert LogNormal.from_mean_cv(mean, cv).quantile(p) > 0.0


class TestSampling:
    @given(
        n=st.integers(min_value=1, max_value=300),
        seed=st.integers(0, 2**32 - 1),
        rho=st.floats(min_value=-0.95, max_value=0.95),
    )
    @settings(max_examples=40)
    def test_correlated_draws_stay_uniform(self, n: int, seed: int, rho: float) -> None:
        drawn = uniforms(np.random.default_rng(seed), n, ("a", "b"), {("a", "b"): rho})
        for values in drawn.values():
            assert values.shape == (n,)
            assert values.min() > 0.0
            assert values.max() < 1.0

    @given(rho=st.floats(min_value=-1.0, max_value=1.0))
    def test_a_two_parameter_matrix_is_always_a_correlation_matrix(self, rho: float) -> None:
        matrix = correlation_matrix(("a", "b"), {("a", "b"): rho})
        assert matrix == pytest.approx(matrix.T)
        assert np.diag(matrix) == pytest.approx(1.0)
        assert float(np.linalg.det(matrix)) == pytest.approx(1.0 - rho**2)

    @given(
        seed=st.integers(0, 2**32 - 1),
        rho=st.floats(min_value=-0.9, max_value=0.9),
    )
    @settings(max_examples=15, deadline=None)
    def test_the_correlation_asked_for_is_the_one_observed(self, seed: int, rho: float) -> None:
        assume(abs(rho) > 0.05)
        drawn = uniforms(np.random.default_rng(seed), 20_000, ("a", "b"), {("a", "b"): rho})
        observed = float(np.corrcoef(drawn["a"], drawn["b"])[0, 1])
        assert observed == pytest.approx(rho, abs=0.05)
