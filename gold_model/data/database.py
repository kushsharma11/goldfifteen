"""Append-only observation store. Market revisions remain separate as-of snapshots."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import JSON, Boolean, Float, Index, Integer, String, create_engine, event, select
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.types import DateTime, TypeDecorator

from gold_model.data.models import BookSnapshot, MarketWindow, PredictionRecord, PricePoint
from gold_model.utils.time import as_utc


class UTCDateTime(TypeDecorator):
    """SQLite drops offsets; store UTC and restore its timezone on every read."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        return None if value is None else as_utc(value).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        from datetime import UTC

        return None if value is None else value.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    pass


class StoredRecord:
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class PriceColumns(StoredRecord):
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    available_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    price: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    provider: Mapped[str] = mapped_column(String)
    contract: Mapped[str | None] = mapped_column(String)
    delayed: Mapped[bool] = mapped_column(Boolean)


class SpotObservation(PriceColumns, Base):
    __tablename__ = "spot_observations"


class ComexObservation(PriceColumns, Base):
    __tablename__ = "comex_observations"


class MarketObservation(StoredRecord, Base):
    __tablename__ = "market_observations"
    market_id: Mapped[str] = mapped_column(String, index=True)
    available_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    start_time: Mapped[datetime] = mapped_column(UTCDateTime)
    end_time: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    target_price: Mapped[float | None] = mapped_column(Float)
    settled_price: Mapped[float | None] = mapped_column(Float)
    result_yes: Mapped[bool | None] = mapped_column(Boolean)
    status: Mapped[str] = mapped_column(String)


class OrderBookObservation(StoredRecord, Base):
    __tablename__ = "orderbook_observations"
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    available_at: Mapped[datetime] = mapped_column(UTCDateTime)
    market_id: Mapped[str] = mapped_column(String, index=True)
    yes_bid: Mapped[float | None] = mapped_column(Float)
    yes_ask: Mapped[float | None] = mapped_column(Float)
    no_bid: Mapped[float | None] = mapped_column(Float)
    no_ask: Mapped[float | None] = mapped_column(Float)
    yes_ask_size: Mapped[float | None] = mapped_column(Float)
    no_ask_size: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (Index("book_market_time", "market_id", "timestamp"),)


class FeatureObservation(StoredRecord, Base):
    __tablename__ = "feature_snapshots"
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    market_id: Mapped[str] = mapped_column(String, index=True)
    seconds_remaining: Mapped[float] = mapped_column(Float)


class ModelPrediction(StoredRecord, Base):
    __tablename__ = "model_predictions"
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    market_id: Mapped[str] = mapped_column(String, index=True)
    model_version: Mapped[str] = mapped_column(String, index=True)
    probability_yes: Mapped[float] = mapped_column(Float)
    probability_no: Mapped[float] = mapped_column(Float)
    kalshi_yes_ask: Mapped[float | None] = mapped_column(Float)
    kalshi_no_ask: Mapped[float | None] = mapped_column(Float)
    yes_edge: Mapped[float | None] = mapped_column(Float)
    no_edge: Mapped[float | None] = mapped_column(Float)
    action: Mapped[str] = mapped_column(String)
    recommended_position: Mapped[int] = mapped_column(Integer)


class RawObservation(StoredRecord, Base):
    __tablename__ = "raw_observations"
    received_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    provider: Mapped[str] = mapped_column(String, index=True)
    kind: Mapped[str] = mapped_column(String)


def _fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


class Database:
    def __init__(self, url: str = "sqlite:///data/gold_model.db"):
        parsed = make_url(url)
        if parsed.drivername.startswith("sqlite") and parsed.database not in (None, ":memory:"):
            Path(parsed.database).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(url)
        if parsed.drivername.startswith("sqlite"):
            @event.listens_for(self.engine, "connect")
            def sqlite_pragmas(connection: Any, _: Any) -> None:
                cursor = connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=5000")
                cursor.close()

    def init(self) -> None:
        Base.metadata.create_all(self.engine)

    def close(self) -> None:
        self.engine.dispose()

    def _append(self, table: type[Base], payload: dict, **columns: Any) -> bool:
        fingerprint = _fingerprint(payload)
        with Session(self.engine) as session, session.begin():
            if session.scalar(select(table.id).where(table.fingerprint == fingerprint)) is not None:
                return False
            session.add(table(fingerprint=fingerprint, payload=payload, **columns))
        return True

    def raw_sink(self, provider: str, kind: str, payload: dict, received_at: datetime) -> None:
        self._append(RawObservation, {
            "provider": provider, "kind": kind,
            "received_at": as_utc(received_at).isoformat(), "response": payload,
        }, provider=provider, kind=kind, received_at=received_at)

    def add_price(self, point: PricePoint, kind: Literal["spot", "comex"] = "spot") -> bool:
        if kind not in ("spot", "comex"):
            raise ValueError("Price kind must be spot or comex")
        table = SpotObservation if kind == "spot" else ComexObservation
        return self._append(table, point.model_dump(mode="json"), **{
            name: getattr(point, name) for name in (
                "timestamp", "available_at", "price", "confidence", "provider", "contract", "delayed"
            )
        })

    def add_market(self, market: MarketWindow) -> bool:
        return self._append(MarketObservation, market.model_dump(mode="json"), **{
            name: getattr(market, name) for name in (
                "market_id", "available_at", "start_time", "end_time", "target_price",
                "settled_price", "result_yes", "status"
            )
        })

    def add_book(self, book: BookSnapshot) -> bool:
        return self._append(OrderBookObservation, book.model_dump(mode="json"), **{
            name: getattr(book, name) for name in (
                "timestamp", "available_at", "market_id", "yes_bid", "yes_ask", "no_bid", "no_ask",
                "yes_ask_size", "no_ask_size"
            )
        })

    def add_features(self, timestamp: datetime, market_id: str, features: dict) -> bool:
        return self._append(FeatureObservation, {
            "timestamp": as_utc(timestamp).isoformat(), "market_id": market_id, "features": features,
        }, timestamp=timestamp, market_id=market_id, seconds_remaining=features["seconds_remaining"])

    def add_prediction(self, prediction: PredictionRecord) -> bool:
        return self._append(ModelPrediction, prediction.model_dump(mode="json"), **{
            name: getattr(prediction, name) for name in (
                "timestamp", "market_id", "model_version", "probability_yes", "probability_no",
                "kalshi_yes_ask", "kalshi_no_ask", "yes_edge", "no_edge", "action", "recommended_position"
            )
        })

    def prices(self, kind: Literal["spot", "comex"] = "spot", *,
               start: datetime | None = None, end: datetime | None = None) -> list[PricePoint]:
        table = SpotObservation if kind == "spot" else ComexObservation
        query = select(table).order_by(table.timestamp, table.available_at, table.id)
        if start is not None:
            query = query.where(table.timestamp >= as_utc(start))
        if end is not None:
            query = query.where(table.timestamp <= as_utc(end))
        with Session(self.engine) as session:
            return [PricePoint.model_validate(row.payload) for row in session.scalars(query)]

    def markets(self, market_id: str | None = None) -> list[MarketWindow]:
        query = select(MarketObservation).order_by(MarketObservation.available_at, MarketObservation.id)
        if market_id:
            query = query.where(MarketObservation.market_id == market_id)
        with Session(self.engine) as session:
            return [MarketWindow.model_validate(row.payload) for row in session.scalars(query)]

    def books(self, market_id: str | None = None, *,
              start: datetime | None = None, end: datetime | None = None) -> list[BookSnapshot]:
        table = OrderBookObservation
        query = select(table).order_by(table.timestamp, table.available_at, table.id)
        if market_id:
            query = query.where(table.market_id == market_id)
        if start is not None:
            query = query.where(table.timestamp >= as_utc(start))
        if end is not None:
            query = query.where(table.timestamp <= as_utc(end))
        with Session(self.engine) as session:
            return [BookSnapshot.model_validate(row.payload) for row in session.scalars(query)]

    def pending_markets(self, at: datetime) -> list[MarketWindow]:
        latest = {market.market_id: market for market in self.markets()}
        return [m for m in latest.values() if m.result_yes is None and m.end_time <= as_utc(at)]
