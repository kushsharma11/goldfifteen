"""Cash-only, integer-contract sizing with fees and exposure caps."""

from dataclasses import dataclass
from math import floor, isfinite

from gold_model.trading.edge import CostConfig, Signal


@dataclass(frozen=True)
class SizingConfig:
    method: str = "fractional_kelly"
    fixed_contracts: int = 1
    fixed_dollar_risk: float = 10.0
    kelly_fraction: float = 0.25
    max_position_dollars: float = 100.0
    max_fraction_of_bankroll_per_market: float = 0.02

    def __post_init__(self) -> None:
        if self.method not in {"fractional_kelly", "fixed_contracts", "fixed_dollar_risk"}:
            raise ValueError("Unknown position-sizing method")
        if self.fixed_contracts < 0 or int(self.fixed_contracts) != self.fixed_contracts:
            raise ValueError("Fixed contracts must be a nonnegative integer")
        if any(
            not isfinite(v) or v < 0 for v in (self.fixed_dollar_risk, self.max_position_dollars)
        ):
            raise ValueError("Dollar limits must be finite and nonnegative")
        if (
            not 0 <= self.kelly_fraction <= 1
            or not 0 <= self.max_fraction_of_bankroll_per_market <= 1
        ):
            raise ValueError("Bankroll and Kelly fractions must lie between zero and one")


@dataclass(frozen=True)
class PositionSize:
    contracts: int
    risk_dollars: float
    execution_price: float | None = None
    fee_dollars: float = 0.0


def size_position(
    signal: Signal,
    bankroll: float,
    costs: CostConfig | None = None,
    config: SizingConfig | None = None,
    available_depth: float | None = None,
) -> PositionSize:
    """Bankroll means currently available cash, excluding unsettled positions.

    Kelly uses the conservative one-contract all-in price. The actual order
    fee is then recomputed, so rounding can never break the cash/risk cap.
    """
    if not isfinite(bankroll) or bankroll < 0:
        raise ValueError("Bankroll must be finite and nonnegative")
    if signal.action == "PASS" or bankroll == 0:
        return PositionSize(0, 0.0)
    costs, config = costs or CostConfig(), config or SizingConfig()
    if signal.selected_ask is None:
        return PositionSize(0, 0.0)
    price = costs.execution_price(signal.selected_ask)
    if not 0 < price < 1:
        return PositionSize(0, 0.0)
    conservative_cost = price + costs.fee(1, price)
    if conservative_cost >= 1:
        return PositionSize(0, 0.0)
    cap = min(
        bankroll, config.max_position_dollars, bankroll * config.max_fraction_of_bankroll_per_market
    )
    if config.method == "fixed_contracts":
        desired = config.fixed_contracts
    else:
        if config.method == "fixed_dollar_risk":
            cap = min(cap, config.fixed_dollar_risk)
        else:
            kelly = max(
                0.0, (signal.selected_probability - conservative_cost) / (1 - conservative_cost)
            )
            cap = min(cap, bankroll * config.kelly_fraction * kelly)
        desired = floor(cap / conservative_cost)
    contracts = min(desired, floor(cap / conservative_cost))
    if available_depth is not None:
        contracts = (
            min(contracts, max(0, floor(available_depth))) if isfinite(available_depth) else 0
        )
    while contracts > 0 and contracts * price + costs.fee(contracts, price) > cap + 1e-10:
        contracts -= 1
    fee = costs.fee(contracts, price)
    return PositionSize(contracts, contracts * price + fee, price, fee)
