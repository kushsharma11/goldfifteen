import pytest

from gold_model.trading.edge import CostConfig, calculate_signal
from gold_model.trading.sizing import SizingConfig, size_position


def test_edge_prices_are_dollars_and_costs_apply_to_threshold():
    costs = CostConfig(fee_per_contract=0.02, slippage_cents=1)
    signal = calculate_signal(.7, .61, .41, .05, costs)
    assert signal.yes_edge == pytest.approx(.09)
    assert signal.yes_adjusted_edge == pytest.approx(.06)
    assert signal.no_edge == pytest.approx(-.11)
    assert signal.action == "YES"
    assert calculate_signal(.7, .61, .41, .07, costs).action == "PASS"


def test_no_side_and_pass_with_stale_or_missing_prices():
    costs = CostConfig(include_fees=False)
    assert calculate_signal(.2, .65, .4, costs=costs).action == "NO"
    assert calculate_signal(.7, None, None, costs=costs).action == "PASS"
    assert calculate_signal(.7, .4, .7, costs=costs, actionable=False).action == "PASS"
    assert calculate_signal(.5, .5, .5, min_edge=0, costs=costs).action == "PASS"


def test_order_fee_rounds_up_once_at_order_level():
    costs = CostConfig(quadratic_fee_rate=.07)
    assert costs.fee(1, .5) == .02
    assert costs.fee(10, .5) == .18
    assert costs.fee(10, .5) < 10 * costs.fee(1, .5)


@pytest.mark.parametrize("method", ["fixed_contracts", "fixed_dollar_risk", "fractional_kelly"])
def test_sizing_obeys_cash_exposure_and_actual_depth(method):
    costs = CostConfig(fee_per_contract=.02)
    signal = calculate_signal(.95, .4, .7, costs=costs)
    config = SizingConfig(method=method, fixed_contracts=100, fixed_dollar_risk=100,
                          max_position_dollars=50, max_fraction_of_bankroll_per_market=.1)
    position = size_position(signal, 100, costs, config, available_depth=3.9)
    assert position.contracts == 3
    assert position.risk_dollars == pytest.approx(1.26)
    assert position.risk_dollars <= min(100, 50, 100 * .1)


def test_pass_and_unaffordable_signal_never_allocate():
    passed = calculate_signal(.5, .6, .6)
    assert size_position(passed, 1000).contracts == 0
    signal = calculate_signal(.9, .5, .6)
    assert size_position(signal, .1).contracts == 0
    assert size_position(signal, 100, available_depth=float("nan")).contracts == 0
    with pytest.raises(ValueError, match="Bankroll"):
        size_position(signal, -1)


def test_quarter_kelly_formula_and_zero_edge():
    costs = CostConfig(include_fees=False)
    config = SizingConfig(kelly_fraction=.25, max_fraction_of_bankroll_per_market=1, max_position_dollars=10000)
    signal = calculate_signal(.7, .5, .6, costs=costs)
    size = size_position(signal, 1000, costs, config)
    # Full Kelly cash fraction (.7-.5)/(1-.5)=.4; quarter Kelly risks $100.
    assert size.risk_dollars == pytest.approx(99.5, abs=.5)
    assert size.risk_dollars <= 100
