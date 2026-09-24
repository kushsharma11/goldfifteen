"""Signals only: all prices, fees, and edges are USD per $1-payoff contract."""

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from math import isfinite
from typing import Literal


@dataclass(frozen=True)
class CostConfig:
    include_fees: bool = True
    fee_per_contract: float = 0.02
    quadratic_fee_rate: float | None = None
    slippage_cents: float = 0.0

    def __post_init__(self) -> None:
        for value in (self.fee_per_contract, self.slippage_cents):
            if not isfinite(value) or value < 0:
                raise ValueError("Costs must be finite and nonnegative")
        if self.quadratic_fee_rate is not None and (
            not isfinite(self.quadratic_fee_rate) or self.quadratic_fee_rate < 0
        ):
            raise ValueError("Fee rate must be finite and nonnegative")

    def fee(self, contracts: int, price: float) -> float:
        """Order-level fee rounded UP to the nearest cent.

        A configured quadratic rate replaces the conservative flat fee. Rates
        are research inputs, not a claim about the market's current fee schedule.
        """
        if contracts < 0 or not isfinite(price) or not 0 <= price <= 1:
            raise ValueError("Invalid fee quantity or price")
        if not self.include_fees or contracts == 0:
            return 0.0
        n, p = Decimal(contracts), Decimal(str(price))
        amount = n * Decimal(str(self.fee_per_contract))
        if self.quadratic_fee_rate is not None:
            amount = Decimal(str(self.quadratic_fee_rate)) * n * p * (1 - p)
        return float(amount.quantize(Decimal("0.01"), rounding=ROUND_CEILING))

    def execution_price(self, ask: float) -> float:
        return ask + self.slippage_cents / 100


@dataclass(frozen=True)
class Signal:
    action: Literal["YES", "NO", "PASS"]
    probability_yes: float
    yes_ask: float | None
    no_ask: float | None
    yes_edge: float | None
    no_edge: float | None
    yes_adjusted_edge: float | None
    no_adjusted_edge: float | None
    reason: str

    @property
    def selected_probability(self) -> float:
        return self.probability_yes if self.action == "YES" else 1 - self.probability_yes

    @property
    def selected_ask(self) -> float | None:
        return self.yes_ask if self.action == "YES" else self.no_ask

    @property
    def selected_edge(self) -> float | None:
        return self.yes_adjusted_edge if self.action == "YES" else self.no_adjusted_edge


def _ask(value: float | None) -> float | None:
    if value is None or not isfinite(value):
        return None
    if not 0 < value < 1:
        return None
    return float(value)


def calculate_signal(
    probability_yes: float,
    yes_ask: float | None,
    no_ask: float | None,
    min_edge: float = 0.05,
    costs: CostConfig | None = None,
    actionable: bool = True,
    reason: str | None = None,
) -> Signal:
    if not isfinite(probability_yes) or not 0 <= probability_yes <= 1:
        raise ValueError("Probability must be finite and between zero and one")
    if not isfinite(min_edge) or min_edge < 0 or min_edge > 1:
        raise ValueError("Minimum edge must be between zero and one")
    costs = costs or CostConfig()
    yes, no = _ask(yes_ask), _ask(no_ask)
    edges: list[float | None] = []
    adjusted: list[float | None] = []
    for probability, ask in ((probability_yes, yes), (1 - probability_yes, no)):
        edges.append(None if ask is None else probability - ask)
        price = None if ask is None else costs.execution_price(ask)
        adjusted.append(
            None if price is None or price >= 1 else probability - price - costs.fee(1, price)
        )
    action = "PASS"
    explanation = reason or "No executable price or insufficient adjusted edge"
    eligible = [
        (value, side)
        for value, side in zip(adjusted, ("YES", "NO"), strict=True)
        if value is not None and value > 0 and value >= min_edge
    ]
    if actionable and eligible:
        _, action = max(eligible)
        explanation = "Adjusted edge meets threshold"
    elif not actionable:
        explanation = reason or "Market data is not actionable"
    return Signal(action, probability_yes, yes, no, *edges, *adjusted, explanation)
