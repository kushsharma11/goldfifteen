"""Validated environment configuration. No credentials are needed for offline research."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    database_url: str = "sqlite:///data/gold_model.db"
    pyth_api_key: SecretStr | None = None
    pyth_base_url: str = "https://pyth.dourolabs.app/hermes"
    pyth_feed_id: str | None = None
    kalshi_api_key: SecretStr | None = None
    kalshi_private_key_path: Path | None = None
    kalshi_base_url: str = "https://external-api.kalshi.com/trade-api/v2"
    kalshi_series_ticker: str = "KXGOLD15M"
    market_reference_verified: bool = False
    comex_provider: Literal["databento", "databento_delayed", "csv", "disabled"] = "databento"
    comex_api_key: SecretStr | None = None
    comex_contract: str | None = None
    comex_csv_path: Path | None = None
    comex_dataset: str = "GLBX.MDP3"
    databento_base_url: str = "https://hist.databento.com/v0"
    comex_delay_seconds: float = Field(default=600, ge=0)
    request_timeout_seconds: float = Field(default=20, gt=0)
    request_retries: int = Field(default=3, ge=0, le=8)
    provider_min_interval_seconds: float = Field(default=1, ge=0)
    historical_latency_seconds: float | None = Field(default=None, ge=0)
    poll_seconds: float = Field(default=5, ge=1)
    spot_max_age_seconds: float = Field(default=15, gt=0)
    book_max_age_seconds: float = Field(default=15, gt=0)
    comex_max_age_seconds: float = Field(default=30, gt=0)
    min_edge: float = Field(default=0.05, ge=0, le=1)
    include_fees: bool = True
    fee_per_contract: float = Field(default=0.02, ge=0, lt=1)
    quadratic_fee_rate: float | None = Field(default=None, ge=0)
    slippage_cents: float = Field(default=0, ge=0, lt=100)
    sizing_method: Literal["fixed_contracts", "fixed_dollar_risk", "fractional_kelly"] = "fractional_kelly"
    kelly_fraction: float = Field(default=0.25, gt=0, le=1)
    max_position_dollars: float = Field(default=100, gt=0)
    max_fraction_of_bankroll_per_market: float = Field(default=0.02, gt=0, le=1)
    fixed_contracts: int = Field(default=1, ge=0)
    fixed_dollar_risk: float = Field(default=10, ge=0)
    bankroll: float = Field(default=1000, gt=0)
    dataset_path: Path = Path("artifacts/dataset.csv")
    model_path: Path = Path("artifacts/logistic.joblib")
    report_dir: Path = Path("artifacts/reports")
    snapshot_seconds: int = Field(default=60, ge=1, lt=900)
    log_level: str = "INFO"

    @field_validator("pyth_base_url", "kalshi_base_url", "databento_base_url")
    @classmethod
    def validate_api_url(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("Provider URLs must use HTTPS")
        return value.rstrip("/")
