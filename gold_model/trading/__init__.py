"""Research signals and sizing recommendations; no order-placement interface."""

from gold_model.trading.edge import CostConfig, Signal, calculate_signal
from gold_model.trading.sizing import PositionSize, SizingConfig, size_position

__all__ = ["CostConfig", "Signal", "calculate_signal", "PositionSize", "SizingConfig", "size_position"]
