# Version 1 validation

Local environment: Python 3.12.13 on macOS ARM64, installed from `uv.lock`.

Completed checks:

- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run pytest -q`: **93 passed**.
- The offline end-to-end test initializes an isolated database, builds causal datasets, trains baseline/logistic models, writes metadata, evaluates held-out probabilities, and runs held-out and walk-forward simulations through the actual Typer CLI.
- Serialization preserves model predictions. Modifying final-test labels cannot affect fitting/calibration. Tests cover source/availability times, delayed/missing feeds, future ticks and metadata, whole-market splits, official labels, sizing caps, fees, both settlement sides, reserved capital, depth, and stale books.
- Provider fixtures test documented response parsing, raw-before-normalization storage, signed authentication, retry behavior, exact contracts, and missing credentials. The live SDK is exercised through a test double; that verifies integration logic rather than actual exchange entitlement or live delivery.
- A real public Kalshi request discovered an active 15-minute gold market and stored metadata. Book retrieval reported the required missing API key/private key; the command returned a failure status while preserving successful observations.

Not verified without external accounts/data:

- Authenticated Pyth delivery/history, authenticated Kalshi order books, and subscribed Databento live/history retrieval.
- Settlement-source/candle parity for the chosen Pyth feed and current series rules.
- Real historical out-of-sample calibration or trading profitability. The repository contains no fabricated performance demonstration.

Operational notes:

Public historical endpoints cannot reconstruct old executable book depth. Strict availability rules intentionally reject backfilled observations that were not recorded at the time. Explicit assumed-latency research is marked in dataset/report provenance and remains weaker evidence than prospective capture. Live collector SQLite throughput and API entitlements should be measured with the actual subscription before relying on long captures.
