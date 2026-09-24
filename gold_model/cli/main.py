"""Gold research CLI. Commands retrieve data and recommend positions, never orders."""

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from gold_model.config import Settings
from gold_model.data.artifacts import json_value, write_dataset
from gold_model.data.database import Database
from gold_model.data.dataset import build_dataset as make_dataset
from gold_model.features.builder import FeatureUnavailable
from gold_model.models.baseline import BaselineModel
from gold_model.services import research
from gold_model.services.collector import Collector
from gold_model.services.predictor import LivePrediction, Predictor, feature_builder
from gold_model.utils.logging import configure_logging
from gold_model.utils.time import parse_datetime

app = typer.Typer(no_args_is_help=True, help="Research Kalshi gold probabilities. No real order execution.")
console = Console()


def settings() -> Settings:
    result = Settings()
    configure_logging(result.log_level)
    return result


def database(config: Settings) -> Database:
    db = Database(config.database_url)
    db.init()
    return db


def show(value: Any) -> None:
    console.print_json(json.dumps(json_value(value), allow_nan=False))


def guarded(operation: Callable[[], Any]) -> Any:
    try:
        return operation()
    except (ValueError, RuntimeError, OSError) as exc:
        console.print(f"[red]Unable to complete:[/red] {exc}", markup=False)
        raise typer.Exit(1) from None


@app.command("init-db")
def init_db() -> None:
    """Create the local append-only observation tables."""
    def run():
        config = settings()
        db = database(config)
        db.close()
        console.print("Database initialized.")
    guarded(run)


@app.command()
def collect(source: Annotated[str, typer.Argument(help="pyth, kalshi, comex, or all")],
            continuous: Annotated[bool, typer.Option("--continuous", help="Keep polling until interrupted")] = False,
            cycles: Annotated[int, typer.Option(min=1, help="Number of cycles when not continuous")] = 1) -> None:
    """Collect and preserve raw responses plus normalized observations."""
    async def run():
        config = settings()
        db = database(config)
        collector = Collector(config, db)
        errors = False
        try:
            iteration = 0
            while continuous or iteration < cycles:
                result = await collector.collect_once(source)
                show(result)
                errors = any(item.get("error") or item.get("errors") for item in result.values())
                iteration += 1
                if continuous or iteration < cycles:
                    await asyncio.sleep(config.poll_seconds)
        finally:
            await collector.aclose()
            db.close()
        if errors:
            raise ValueError("One or more sources failed; successfully received observations were preserved")
    guarded(lambda: asyncio.run(run()))


@app.command()
def backfill(source: Annotated[str, typer.Argument(help="pyth, kalshi, or comex")],
             start: Annotated[str, typer.Option(help="Inclusive ISO datetime with UTC offset")],
             end: Annotated[str, typer.Option(help="Exclusive ISO datetime with UTC offset")],
             step_seconds: Annotated[int, typer.Option(min=1)] = 5) -> None:
    """Retrieve documented historical data; does not invent historical order books."""
    async def run():
        config = settings()
        if source not in {"pyth", "kalshi", "comex"}:
            raise ValueError("Backfill source must be pyth, kalshi, or comex")
        db = database(config)
        collector = Collector(config, db)
        try:
            show(await collector.backfill(source, parse_datetime(start), parse_datetime(end), step_seconds))
        finally:
            await collector.aclose()
            db.close()
    guarded(lambda: asyncio.run(run()))


@app.command("build-dataset")
def build_dataset(snapshot_seconds: Annotated[int | None, typer.Option(min=1, max=899)] = None,
                  output: Path | None = None) -> None:
    """Generate causal feature snapshots and official settlement labels."""
    def run():
        config = settings()
        db = database(config)
        try:
            frame = make_dataset(db.markets(), db.prices("spot"), db.prices("comex"), db.books(),
                                 snapshot_seconds=snapshot_seconds or config.snapshot_seconds,
                                 builder=feature_builder(config))
            path = output or config.dataset_path
            write_dataset(frame, path, {
                "availability_policy": "source timestamp AND availability timestamp <= prediction timestamp",
                "historical_latency_seconds": config.historical_latency_seconds,
                "settlement_labels": "official Kalshi outcomes",
                "market_reference_verified": config.market_reference_verified,
                "snapshot_seconds": snapshot_seconds or config.snapshot_seconds,
            })
            show({"path": path, "rows": len(frame), "markets": frame.market_id.nunique(),
                  "skipped": len(frame.attrs.get("skipped", [])), "manifest": path.with_suffix(".manifest.json")})
            if frame.empty:
                raise ValueError("No eligible snapshots. Collect open-market metadata, at least five minutes of prices, and later official settlements; inspect the manifest.")
        finally:
            db.close()
    guarded(run)


@app.command()
def train(model: Annotated[str, typer.Argument(help="baseline or logistic")],
          features: Annotated[str | None, typer.Option(help="Comma-separated feature list for ablation")] = None,
          calibration: Annotated[str, typer.Option(help="sigmoid, isotonic, or none")] = "sigmoid") -> None:
    """Train on chronological market groups, calibrate later, evaluate held-out last."""
    guarded(lambda: show(research.train(settings(), model,
                                        features=features.split(",") if features else None,
                                        calibration=calibration)))


@app.command()
def evaluate(model: str = "logistic") -> None:
    """Compare held-out Brier/log loss and reliability with baseline and Kalshi."""
    guarded(lambda: show(research.evaluate(settings(), model)))


@app.command()
def backtest(model: str = "logistic", walk_forward: bool = False, idealized: bool = False,
             bankroll: Annotated[float | None, typer.Option(min=0.01)] = None,
             min_train_markets: Annotated[int, typer.Option(min=2)] = 20,
             calibration_markets: Annotated[int, typer.Option(min=2)] = 5,
             test_markets: Annotated[int, typer.Option(min=1)] = 5,
             features: str | None = None) -> None:
    """Simulate OOS signals with asks, depth, fees, slippage, and reserved cash."""
    def run():
        config = settings()
        if bankroll is not None:
            config = config.model_copy(update={"bankroll": bankroll})
        show(research.backtest(config, model, walk_forward=walk_forward, idealized=idealized,
                              min_train_markets=min_train_markets, calibration_markets=calibration_markets,
                              test_markets=test_markets, features=features.split(",") if features else None))
    guarded(run)


def prediction_panel(result: LivePrediction, min_edge: float) -> Panel:
    f, r = result.features, result.record
    table = Table.grid(padding=(0, 3))
    table.add_column(style="dim")
    table.add_column(justify="right")

    def row(label: str, value: float | None, fmt: str) -> None:
        table.add_row(label, "unavailable" if value is None else format(value, fmt))

    table.add_row("Market", r.market_id)
    row("Target ($/oz)", f["target_price"], ",.2f")
    row("Current Pyth Gold ($/oz)", f["current_gold_price"], ",.2f")
    row("Distance ($/oz)", f["distance_from_target"], "+.2f")
    seconds = int(f["seconds_remaining"])
    table.add_row("Time Remaining", f"{seconds // 60:02d}:{seconds % 60:02d}")
    for name, key in (("Gold 30s", "gold_return_30s"), ("Gold 1m", "gold_return_1m"), ("Gold 3m", "gold_return_3m"), ("COMEX 1m", "comex_return_1m")):
        row(name, f[key], "+.3%")
    row("5m volatility ($/√second)", f["volatility_5m"], ".5f")
    row("Expected remaining move ($)", f["expected_remaining_move"], ".3f")
    row("Baseline YES", f["baseline_probability"], ".1%")
    row("Model YES", r.probability_yes, ".1%")
    row("Model NO", r.probability_no, ".1%")
    for side in ("yes", "no"):
        ask = getattr(r, f"kalshi_{side}_ask")
        table.add_row(f"Kalshi {side.upper()} Ask", "unavailable" if ask is None else f"{ask * 100:.2f}¢")
        row(f"{side.upper()} Edge", getattr(r, f"{side}_edge"), "+.1%")
    row("Minimum adjusted edge", min_edge, ".1%")
    table.add_row("Signal", r.action)
    table.add_row("Recommended position", f"{r.recommended_position} contracts / ${r.risk_dollars:.2f} risk")
    table.add_row("Reason", r.reason)
    table.add_row("Model", r.model_version)
    return Panel(table, title="KALSHI GOLD 15M", subtitle="Research recommendations · no execution")


async def live_predictions(model: str, bankroll: float | None, refresh: bool,
                           cycles: int, market_id: str | None, monitor: bool) -> None:
    config = settings()
    if model not in {"baseline", "logistic"}:
        raise ValueError("Live model must be baseline or logistic")
    selected = BaselineModel() if model == "baseline" else research.load_model(config.model_path)
    db = database(config)
    collector = Collector(config, db)
    predictor = Predictor(config, db, selected)

    async def cycle():
        if refresh:
            await collector.collect_once("all")
        try:
            result = predictor.predict(bankroll=bankroll, market_id=market_id)
            return prediction_panel(result, config.min_edge)
        except (FeatureUnavailable, ValueError) as exc:
            if not monitor:
                raise
            return Panel(f"NO SIGNAL — {exc}", title="KALSHI GOLD 15M")

    try:
        if not monitor:
            console.print(await cycle())
        else:
            with Live(Panel("Collecting observations…", title="KALSHI GOLD 15M"), console=console, refresh_per_second=1) as live:
                iteration = 0
                while cycles == 0 or iteration < cycles:
                    live.update(await cycle(), refresh=True)
                    iteration += 1
                    if cycles == 0 or iteration < cycles:
                        await asyncio.sleep(config.poll_seconds)
    finally:
        await collector.aclose()
        db.close()


@app.command()
def predict(model: str = "baseline", bankroll: Annotated[float | None, typer.Option(min=0.01)] = None,
            refresh: bool = False, market_id: str | None = None) -> None:
    """Predict from stored observations; optionally refresh all feeds first."""
    guarded(lambda: asyncio.run(live_predictions(model, bankroll, refresh, 1, market_id, False)))


@app.command()
def monitor(model: str = "baseline", bankroll: Annotated[float | None, typer.Option(min=0.01)] = None,
            cycles: Annotated[int, typer.Option(min=0, help="0 runs until interrupted")] = 0,
            market_id: str | None = None) -> None:
    """Collect continuously, display probabilities and log bounded recommendations."""
    guarded(lambda: asyncio.run(live_predictions(model, bankroll, True, cycles, market_id, True)))


if __name__ == "__main__":
    app()
