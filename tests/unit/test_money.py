"""Money carries a magnitude and a direction, and refuses to carry a sign."""

from __future__ import annotations

import numpy as np
import pytest

from simplyinvest.errors import SignConventionError
from simplyinvest.money import Amount, Direction


class TestDirection:
    def test_signs(self):
        assert Direction.OUT.sign == -1.0
        assert Direction.IN.sign == 1.0


class TestConstruction:
    def test_paid_is_an_outflow(self):
        assert Amount.paid(23_900).signed == -23_900.0

    def test_received_is_an_inflow(self):
        assert Amount.received(2_400).signed == 2_400.0

    def test_zero_has_no_effect_either_way(self):
        assert Amount.zero().signed == 0.0

    def test_a_negative_magnitude_is_refused(self):
        with pytest.raises(SignConventionError, match="direction"):
            Amount.paid(-500)

    def test_the_refusal_names_the_way_out(self):
        with pytest.raises(SignConventionError, match=r"Amount\.received\(500\.0\)"):
            Amount.paid(-500)

    def test_a_nan_is_refused(self):
        with pytest.raises(SignConventionError, match="finite"):
            Amount.paid(float("nan"))

    @pytest.mark.parametrize(
        ("signed", "expected"),
        [(-500.0, Direction.OUT), (500.0, Direction.IN), (0.0, Direction.IN)],
    )
    def test_net_reads_the_direction_off_the_sign(self, signed, expected):
        assert Amount.net(signed).direction is expected

    def test_net_of_a_negative_scalar_round_trips(self):
        assert Amount.net(-1_250.0).signed == -1_250.0

    def test_net_of_a_uniform_batch_takes_that_direction(self):
        assert Amount.net(np.array([-5.0, -3.0])).direction is Direction.OUT
        assert Amount.net(np.array([5.0, 3.0])).direction is Direction.IN

    def test_net_refuses_a_batch_that_spans_zero(self):
        with pytest.raises(SignConventionError, match="spans zero"):
            Amount.net(np.array([-5.0, 3.0]))


class TestBatch:
    def test_an_array_magnitude_is_accepted(self):
        amount = Amount.paid(np.array([1.0, 2.0, 3.0]))
        assert np.array_equal(np.asarray(amount.signed), np.array([-1.0, -2.0, -3.0]))
        assert not amount.is_scalar

    def test_a_negative_anywhere_in_the_batch_is_refused(self):
        with pytest.raises(SignConventionError):
            Amount.paid(np.array([1.0, -2.0]))


class TestOperations:
    def test_scaling_keeps_the_direction(self):
        assert Amount.paid(100).scaled(0.5).signed == -50.0

    def test_reversing_flips_it(self):
        assert Amount.paid(100).reversed().signed == 100.0

    def test_str_shows_the_direction(self):
        assert str(Amount.paid(23_900)) == "-23,900.00"
        assert str(Amount.received(2_400)) == "+2,400.00"

    def test_amounts_compare_by_value(self):
        assert Amount.paid(100) == Amount.paid(100)
        assert Amount.paid(100) != Amount.received(100)
