"""Normalized records. Monetary units: USD per ounce or USD per $1 binary contract."""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gold_model.utils.time import as_utc

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
PositivePrice = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class Observation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    @field_validator("*", mode="after")
    @classmethod
    def normalize_datetimes(cls, value: Any) -> Any:
        return as_utc(value) if isinstance(value, datetime) else value


class PricePoint(Observation):
    timestamp: datetime
    available_at: datetime
    price: PositivePrice
    provider: str
    confidence: float | None = Field(default=None, ge=0)
    feed_id: str | None = None
    contract: str | None = None
    delayed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def availability(self) -> "PricePoint":
        if self.available_at < self.timestamp:
            raise ValueError("Price cannot be available before its source timestamp")
        return self


class MarketWindow(Observation):
    market_id: str
    event_id: str | None = None
    start_time: datetime
    end_time: datetime
    available_at: datetime
    target_price: PositivePrice | None = None
    settled_price: PositivePrice | None = None
    result_yes: bool | None = None
    settlement_time: datetime | None = None
    status: str = "open"
    raw: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def interval(self) -> "MarketWindow":
        if self.end_time <= self.start_time:
            raise ValueError("Market end must follow start")
        if self.settlement_time is not None and self.settlement_time < self.end_time:
            raise ValueError("Settlement time cannot precede market end")
        return self


class BookSnapshot(Observation):
    timestamp: datetime
    available_at: datetime
    market_id: str
    yes_bid: Probability | None = None
    yes_ask: Probability | None = None
    no_bid: Probability | None = None
    no_ask: Probability | None = None
    yes_ask_size: float | None = Field(default=None, ge=0)
    no_ask_size: float | None = Field(default=None, ge=0)
    yes_bids: list[list[float]] = Field(default_factory=list)
    no_bids: list[list[float]] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def book_integrity(self) -> "BookSnapshot":
        if self.available_at < self.timestamp:
            raise ValueError("Book cannot be available before its source timestamp")
        for side in ("yes", "no"):
            bid, ask = getattr(self, f"{side}_bid"), getattr(self, f"{side}_ask")
            if bid is not None and ask is not None and bid > ask + 1e-9:
                raise ValueError("Crossed order book")
        for price, size in self.yes_bids + self.no_bids:
            if not 0 <= price <= 1 or size < 0:
                raise ValueError("Invalid book depth")
        return self


class PredictionRecord(Observation):
    timestamp: datetime
    market_id: str
    model_version: str
    probability_yes: Probability
    probability_no: Probability
    kalshi_yes_ask: Probability | None = None
    kalshi_no_ask: Probability | None = None
    yes_edge: float | None = None
    no_edge: float | None = None
    action: Literal["YES", "NO", "PASS"]
    recommended_position: int = Field(ge=0)
    risk_dollars: float = Field(ge=0)
    reason: str = ""

    @model_validator(mode="after")
    def complementary(self) -> "PredictionRecord":
        if abs(self.probability_yes + self.probability_no - 1) > 1e-9:
            raise ValueError("YES and NO probabilities must sum to one")
        if self.action == "PASS" and self.recommended_position:
            raise ValueError("PASS must have zero contracts")
        return self
