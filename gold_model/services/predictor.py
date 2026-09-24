"""Live predictions use the same point-in-time features as research datasets."""

import logging
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

from gold_model.config import Settings
from gold_model.data.database import Database
from gold_model.data.models import PredictionRecord
from gold_model.features.builder import FeatureBuilder, FeatureUnavailable
from gold_model.models.baseline import BaselineModel
from gold_model.trading.edge import CostConfig, Signal, calculate_signal
from gold_model.trading.sizing import PositionSize, SizingConfig, size_position
from gold_model.utils.time import as_utc, utc_now

logger = logging.getLogger(__name__)


def feature_builder(settings: Settings) -> FeatureBuilder:
    return FeatureBuilder(
        spot_max_age_seconds=settings.spot_max_age_seconds,
        book_max_age_seconds=settings.book_max_age_seconds,
        comex_max_age_seconds=settings.comex_max_age_seconds,
    )


def cost_config(settings: Settings) -> CostConfig:
    return CostConfig(
        include_fees=settings.include_fees, fee_per_contract=settings.fee_per_contract,
        quadratic_fee_rate=settings.quadratic_fee_rate, slippage_cents=settings.slippage_cents,
    )


def sizing_config(settings: Settings) -> SizingConfig:
    return SizingConfig(
        method=settings.sizing_method, fixed_contracts=settings.fixed_contracts,
        fixed_dollar_risk=settings.fixed_dollar_risk, kelly_fraction=settings.kelly_fraction,
        max_position_dollars=settings.max_position_dollars,
        max_fraction_of_bankroll_per_market=settings.max_fraction_of_bankroll_per_market,
    )


@dataclass(frozen=True)
class LivePrediction:
    record: PredictionRecord
    features: dict[str, float | None]
    signal: Signal
    position: PositionSize


class Predictor:
    def __init__(self, settings: Settings, database: Database, model: Any = None):
        self.settings = settings
        self.database = database
        self.model = model if model is not None else BaselineModel()
        self.builder = feature_builder(settings)

    def predict(self, *, at: datetime | None = None, market_id: str | None = None,
                bankroll: float | None = None) -> LivePrediction:
        at = as_utc(at) if at is not None else utc_now()
        versions = [m for m in self.database.markets(market_id) if m.available_at <= at]
        latest = {m.market_id: m for m in versions}
        active = [m for m in latest.values() if m.start_time <= at < m.end_time and m.status in {"open", "active"}]
        if not active:
            raise FeatureUnavailable("NO SIGNAL — no currently open Kalshi gold window is available")
        market = min(active, key=lambda m: m.end_time)
        start = at - timedelta(hours=1)
        features = self.builder.build(
            at, market, self.database.prices("spot", start=start, end=at),
            self.database.prices("comex", start=start, end=at),
            self.database.books(market.market_id, start=start, end=at),
        )
        probability = self.model.predict(features)
        costs = cost_config(self.settings)
        signal = calculate_signal(
            probability.probability_yes, features["kalshi_yes_ask"], features["kalshi_no_ask"],
            self.settings.min_edge, costs,
            actionable=self.settings.market_reference_verified,
            reason=None if self.settings.market_reference_verified else "PASS — settlement reference/feed parity has not been verified",
        )
        depth = features.get(f"kalshi_{signal.action.lower()}_ask_size") if signal.action != "PASS" else None
        # Unknown depth cannot support a quantity recommendation.
        position = size_position(signal, self.settings.bankroll if bankroll is None else bankroll,
                                 costs, sizing_config(self.settings), available_depth=depth or 0)
        if signal.action != "PASS" and position.contracts == 0:
            signal = replace(signal, action="PASS", reason="PASS — insufficient depth or risk budget for one contract")
        record = PredictionRecord(
            timestamp=at, market_id=market.market_id, model_version=self.model.version,
            probability_yes=probability.probability_yes, probability_no=probability.probability_no,
            kalshi_yes_ask=signal.yes_ask, kalshi_no_ask=signal.no_ask,
            yes_edge=signal.yes_edge, no_edge=signal.no_edge, action=signal.action,
            recommended_position=position.contracts, risk_dollars=position.risk_dollars, reason=signal.reason,
        )
        self.database.add_features(at, market.market_id, features)
        self.database.add_prediction(record)
        logger.info("model_prediction", extra={"context": record.model_dump(mode="json")})
        return LivePrediction(record, features, signal, position)
