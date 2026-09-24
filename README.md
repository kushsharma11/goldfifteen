# Gold Model

Python 3.12+ research system for Kalshi 15-minute gold markets. Estimates the probability of finishing at or above a market's reference, compares it with executable asks, and recommends bounded positions. **There is no order-placement functionality.**

Implementation checkpoint: adapters, storage, causal features, models, simulation, and CLI are implemented. Integration verification is in progress. No historical profitability is asserted; no production data is bundled.

## Install and first run

```bash
uv sync --python 3.12 --extra comex
cp .env.example .env
# Edit .env with your data credentials and exact futures contract.
uv run gold-model init-db
uv run gold-model collect all
uv run gold-model collect all --continuous
```

Keep collection running through enough resolved windows to build training, calibration, and test periods, with both outcomes in fitting periods. Five minutes of continuous spot history are needed for baseline volatility. For live baseline monitoring, which also collects observations:

```bash
uv run gold-model monitor --model baseline --bankroll 1000
```

Then:

```bash
uv run gold-model build-dataset
uv run gold-model train baseline
uv run gold-model train logistic
uv run gold-model evaluate
uv run gold-model backtest
uv run gold-model backtest --walk-forward
uv run gold-model monitor --model logistic --bankroll 1000
```

`predict` uses stored observations; `predict --refresh` collects first. `monitor --cycles 1` performs a single refresh/display cycle. `collect all --cycles 12` performs a bounded capture. Configuration is environment-driven; `.env` is ignored by Git. Never load untrusted joblib model files: the CLI treats the configured local artifact as trusted, and joblib deserialization can execute Python.

## Data access and limitations

Read [provider documentation](docs/providers.md) for endpoints, schemas, and links to official sources.

- **Pyth:** current Hermes access requires `PYTH_API_KEY`. Set `PYTH_FEED_ID` to the reference feed after verifying the series rules, or allow unambiguous feed discovery. Raw responses, confidence, source timestamps and receipt timestamps are retained.
- **Kalshi:** public market discovery supports the configurable `KALSHI_SERIES_TICKER`. Order-book retrieval requires `KALSHI_API_KEY` and `KALSHI_PRIVATE_KEY_PATH` under the documented API. Historical market results do not include historical books. Capture books prospectively; do not treat candles, last trades, or final quotes as historical executable asks.
- **COMEX:** `COMEX_PROVIDER=databento` uses licensed live Databento data, `COMEX_API_KEY`, and an exact listed `COMEX_CONTRACT`. Install the `comex` extra. Historical/delayed retrieval and a documented CSV import source are supported. Delayed observations are marked and excluded from live momentum features. `COMEX_PROVIDER=disabled` explicitly disables this source; the default logistic feature set does not need it.
- **Settlement:** observed Kalshi gold rules use one-minute Pyth candlesticks, rounded values, and a prior-published-value fallback. The baseline's instantaneous normal diffusion is an approximation. Verify the exact feed, target, comparison/equality, rounding, and candle convention before setting `MARKET_REFERENCE_VERIFIED=true`. Until then, live probabilities are displayed with PASS recommendations. Official Kalshi results supply all training labels.

No automatic source substitution, fake market response, synthetic history, or profitability fallback exists. Empty data and missing required model features produce explicit errors.

## Historical collection

```bash
uv run gold-model backfill pyth --start 2026-09-01T00:00:00Z --end 2026-09-01T01:00:00Z --step-seconds 5
uv run gold-model backfill kalshi --start 2026-09-01T00:00:00Z --end 2026-09-02T00:00:00Z
uv run gold-model backfill comex --start 2026-09-01T00:00:00Z --end 2026-09-01T01:00:00Z
```

Pyth timestamp requests can return an update after the requested instant. The adapter preserves its real publish time; feature generation never moves it backwards. Historical data can be paid and subject to retention, rate limits, and entitlements.

Default availability is actual receipt time. A download today cannot establish what was known in a past window. Consequently, strict retrospective datasets require previously captured metadata/ticks/books. For explicitly assumption-based research, `HISTORICAL_LATENCY_SECONDS` assigns source-time-plus-latency availability to backfilled prices. Historical target metadata then receives a separately marked assumed-open snapshot, stripped of outcome fields; the official result remains a separate version. A final target may differ from its original published value, so this option is not evidence of point-in-time correctness. Dataset manifests preserve these assumptions. No setting fabricates historical books.

## Research design

SQLite stores append-only raw responses, spot/futures ticks, market versions, order-book snapshots, feature snapshots, and versioned predictions. Original response payloads remain available for recomputation. Both source time and availability time must be at or before a feature timestamp. All timestamps are timezone-aware UTC, including after SQLite reads.

Returns use backward observations only and remain missing when cadence, coverage, or contract continuity is inadequate. Realized arithmetic volatility is `sqrt(sum(price_increment²) / elapsed_seconds)`, measured in USD per square-root second. The expected remaining move is that volatility times `sqrt(seconds_remaining)`. The normal baseline is `Phi((spot - target) / expected_remaining_move)` with numerical floors and clipped probabilities. Prices are USD per troy ounce; binary quotes and probabilities are fractions from 0 to 1. Contract P&L is USD for a $1 payout.

The logistic model uses the baseline log-odds, 30-second and one-minute spot returns, and five-minute volatility. Scaling and regularized fitting use the first period; sigmoid or isotonic calibration uses a later period; evaluation uses whole, later, unseen markets. Each market gets equal total fitting weight across its correlated snapshots. This is Approach A: baseline as an input, allowing a learned adjustment without a more complex offset estimator. Calibration-period diagnostics are identified as fit-period diagnostics.

Splits are chronological and grouped by market. Labels unavailable by the next period are purged. Walk-forward fitting repeats train/calibrate/test on expanding history. Missing required inputs fail explicitly. Feature ablation uses `train logistic --features baseline_logit,gold_return_1m`; use validation periods to choose features and reserve a final untouched period.

## Execution assumptions and reports

The simulator buys the recorded ask, caps quantity at visible opposing bid depth, permits at most one entry per market, and reserves entry cash until outcome availability. Stale/missing books yield PASS. `--idealized` is a separate entire-run midpoint simulation with no depth claim; it never fills gaps in an executable run. Even recorded asks do not guarantee a fill after network and decision latency.

Default fees use an explicit conservative flat $0.02 per contract assumption, not a claim about the current exchange fee schedule. Set `FEE_PER_CONTRACT`, or `QUADRATIC_FEE_RATE` for a quadratic order-level fee rounded up to cents, `SLIPPAGE_CENTS`, and `INCLUDE_FEES`. Reports state cost assumptions. Quarter Kelly is the default, bounded by available cash, $100 maximum risk and 2% of bankroll per market. Fixed-contract and fixed-dollar methods are also supported.

`artifacts/reports/` contains JSON probability/trading metrics, reliability-bin observations, strategy comparisons on common support, threshold sweeps, decisions, trades, equity, and out-of-sample predictions. Metrics include Brier score, log loss, calibration, P&L, ROI, settlement-equity drawdown, side/edge/time/volatility breakdowns and an unannualized trade-level risk statistic where supported. Repeated snapshots are correlated; raw row counts are not independent trials. Settled-equity drawdown does not represent intramarket mark-to-market risk.

Threshold sweeps are descriptive. Selecting the best final-test threshold or repeatedly revising rules against its results would invalidate the holdout. The application does not assert that a custom model outperforms Kalshi or that past P&L predicts future returns.

## Development

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

Tests use synthetic fixtures solely for deterministic correctness checks; their results are not market-performance evidence. Tests do not require provider credentials. A network smoke check can use `collect kalshi`; full live verification requires configured keys and subscriptions.

The most valuable next research steps are prospective reference-aligned tick/book capture, out-of-sample volatility and calibration comparisons, and controlled COMEX/Pyth timing ablations with realistic latency.
