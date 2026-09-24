# Provider integration and availability

All providers are read-only. No adapter places orders. Gold prices and Pyth
confidence widths are USD per troy ounce. Kalshi prices are dollars per $1
binary contract, numerically equivalent to probabilities in `[0, 1]`. Book
sizes are contracts and may be fractional. COMEX prices are USD per troy ounce;
the application recommends Kalshi positions, not futures lots.

Every normalized record has an event `timestamp` and a separate `available_at`.
The collector persists the raw response through its `raw_sink` before the
provider normalizes it. Live data is available at local receipt time. A repeated
live trade retains its original availability time, so repeated polling does not
make a stale trade fresh. All timestamps require an explicit timezone and are
normalized to UTC.

## Pyth XAU/USD

Configure `PYTH_API_KEY`. Optional `PYTH_FEED_ID` pins an exact hexadecimal Core
feed identifier; otherwise `/v2/price_feeds` discovers the single `XAU/USD` metal
feed by its exact symbol. Ambiguous matches fail rather than picking a feed.
Default `PYTH_BASE_URL=https://pyth.dourolabs.app/hermes` uses the current Hermes
service with bearer authentication. Pyth's August 26, 2026 upgrade introduced
authentication for all Hermes users. Access depends on the account's plan.

`latest_price()` polls `/v2/updates/price/latest`, applying Pyth's integer price
and confidence exponent. `historical_prices(start, end, step_seconds=5)` samples
the documented `/v2/updates/price/{timestamp}` API. Actual returned publish times
are preserved even when later than the requested timestamp; duplicate updates
are collapsed and no interpolation is performed. Historical retention and
entitlements depend on the service. Unsupported ranges fail explicitly. Long
backfills can be slow: the conservative default pacing is one request per second.
The app does not invent a bulk endpoint or silently use candle closes as earlier
tick observations.

Official sources, checked September 23, 2026:

- [Current Hermes API and authentication](https://docs.pyth.network/price-feeds/core/fetch-price-updates)
- [Core upgrade and API-key requirements](https://docs.pyth.network/price-feeds/core/upgrade/preparing)
- [Historical timestamp API and limits](https://docs.pyth.network/price-feeds/core/use-historical-price-data)
- [Feed catalog](https://docs.pyth.network/price-feeds/core/price-feeds)

## Kalshi gold windows

`KALSHI_SERIES_TICKER=KXGOLD15M` selects a series, never a hard-coded individual
market. The provider checks that ticker prefixes match and each market has
exactly 900 seconds between `open_time` and `close_time`. Unsupported windows
are logged and rejected. A market target is read from `floor_strike` only for
an at-or-above-compatible strike type. Missing targets remain missing; titles
and subtitles are not scraped to fabricate a target. Official `result` supplies
the label. `expiration_value`, when numeric, supplies the underlying settlement
price. **`settlement_value_dollars` is a binary contract payout, not a gold
price**, and is never used as a gold settlement value.

Public market discovery does not require credentials. The documented order-book
endpoint requires `KALSHI_API_KEY` (key ID) and `KALSHI_PRIVATE_KEY_PATH` (an
unencrypted PEM file). Both RSA-PSS/SHA-256 and Ed25519 keys are supported. A new
timestamp/signature is generated for each retry, excluding query parameters
from the signed path. Keys never appear in raw observation storage or logs.

Books use current `orderbook_fp.yes_dollars/no_dollars` values. A legacy cents
schema is accepted only under its explicit legacy field name. Kalshi returns
bids; YES ask is `1 - best NO bid`, and NO ask is `1 - best YES bid`. Ask sizes
come from the opposite bid side. Empty or 404 books produce absent asks and
sizes; a crossed book fails. Book timestamps use the response receipt time:
HTTP does not reveal individual order age or guarantee a future fill.

History combines `/markets` for recent windows and `/historical/markets` for
archived windows. Both are paginated; archived history is filtered locally
because its documented endpoint does not offer time filters. A later fetch of
a resolved market does not prove its final target was available during the
window. Historical order-book depth is not reconstructed from market summaries
or candlesticks. Start collecting books now for execution-oriented research.

The live KXGOLD15M rules inspected during implementation refer to **one-minute
Pyth GOLD candle close values**, settlement rounding to two decimals, and a
fallback to the most recent published value when the specified observation is
missing. These rules matter at ties and around missing ticks. An XAU/USD tick
is not evidence of exact settlement-source parity. Verify the configured feed,
candle boundary convention, and current market rules before setting
`MARKET_REFERENCE_VERIFIED=true`; the default guard prevents an actionable
recommendation until that verification is recorded. Always use official Kalshi
outcomes for supervised labels.

- [Get markets and recent/history split](https://docs.kalshi.com/api-reference/market/get-markets)
- [Get archived markets](https://docs.kalshi.com/api-reference/historical/get-historical-markets)
- [Order-book schema and bid/ask complement](https://docs.kalshi.com/api-reference/market/get-market-orderbook)
- [RSA and Ed25519 authentication](https://docs.kalshi.com/getting_started/api_keys)
- [Official gold 15-minute market](https://kalshi.com/markets/kx/test/kxgold15m-26aug070700)

## COMEX / Databento

Set `COMEX_PROVIDER=databento`, `COMEX_API_KEY`, and an exact listed
`COMEX_CONTRACT`, such as `GCZ6` **only while that expiry is appropriate for your
research dates**. Install the live extra: `pip install -e '.[comex]'`. Real-time
CME/COMEX exchange entitlements and an appropriate Databento plan are required;
there is no assumed universally free real-time source. No API call is made to
purchase a subscription.

The live SDK maintains a `trades` subscription to the exact `raw_symbol` in
`GLBX.MDP3`. Every received trade is queued, raw-persisted, and normalized before
updating the cached latest price. Prices are converted from Databento's `1e-9`
units. Receipt time is recorded locally; `delayed=false` applies to this live
path. Gateway errors and capture-queue overflow disable predictions. The SDK
reconnects dropped sessions, while the application still checks the age of the
last received trade. A quiet/closed market produces a timeout or a stale price.
The SDK is loaded only when live COMEX collection is selected.

Historical retrieval uses the documented **read-only** HTTP RPC
`POST /v0/timeseries.get_range`, HTTP Basic authentication, and JSON-lines
`trades` with explicit pretty prices/timestamps and mapped symbols. Requests are
split into at most one-hour chunks. Historical records are marked delayed and
have download-time availability by default. The raw nanosecond timestamps are
preserved; Python datetime normalization conservatively rounds live nanoseconds
up to microseconds. Continuous symbols and all-contract parent queries are
prohibited. Collect a new expiry separately at roll and never stitch price
levels across contracts into a return calculation.

`COMEX_PROVIDER=databento_delayed` is an explicit historical polling alternative.
It polls up to `COMEX_DELAY_SECONDS` before now (default 600), always marks the
observation delayed, and keeps its actual download availability. The delay must
match your account's availability; this is not a promise that ten-minute-old
data is included in every plan. The live predictor can reject delayed COMEX or
use a separately trained model without COMEX features.

- [Historical HTTP endpoint, schemas, and authentication](https://databento.com/docs/api-reference-historical/timeseries/timeseries-get-range?historical=http)
- [Live subscription, callbacks, sessions and errors](https://databento.com/docs/api-reference-live/client/subscribe)
- [Official live quickstart](https://databento.com/docs/quickstart/build-first-app?historical=python&live=python)

## COMEX / externally captured CSV

`COMEX_PROVIDER=csv` with `COMEX_CSV_PATH` reads an externally maintained CSV.
Its required columns are:

```csv
timestamp,available_at,price,contract,delayed
```

Timestamps must include `Z` or an explicit UTC offset. `available_at` must be
the actual recorded receipt time, at or after the event. Price must be positive.
`contract` must equal the configured exact expiry on every row. `delayed` must
be explicitly `true`/`false` or `1`/`0`. Missing timestamps, mixed contracts and
unknown delay flags fail. Atomically replace the CSV when updating it so readers
never encounter partial writes. The application does not infer receipt times
from file modification times or label an unknown feed real-time.

## Retrospective availability assumptions

Leave `HISTORICAL_LATENCY_SECONDS` unset for the conservative default: remote
backfills are unavailable to any prediction before their actual download.
Consequently, a backfill of settled markets and historical prices alone may
produce no valid retrospective training rows. That is intentional evidence of
missing historical availability, not a reason to bypass the guard silently.

An explicit nonnegative value opts into an **assumed** historical availability
model: Pyth uses publish time plus latency, Databento uses the later of event and
capture-server receive time plus latency, and a sanitized Kalshi opening target
version uses market open plus latency. The opening version has no outcome or
settlement fields; the final version is separate. A final label is available at
official `settlement_ts` plus assumed latency when that timestamp exists, and at
download time otherwise. This assumption cannot prove unrevised target availability.
It is recorded as `availability_basis=assumed_historical_latency` with download
time and the latency in price metadata and `market.raw._availability`. Research
using it must be identified as retrospective assumption-based research, and
must not be described as verified executable replay. Independently captured
point-in-time books and metadata are the preferred source.

## Failure handling and limits

HTTP transport errors, rate limits, and 5xx responses receive bounded retries.
`Retry-After` is honored; requests asking for a pause longer than sixty seconds
fail with a clear retry message. Authorization and malformed data failures are
not masked by synthetic fallbacks. `REQUEST_TIMEOUT_SECONDS`,
`REQUEST_RETRIES`, and `PROVIDER_MIN_INTERVAL_SECONDS` control pacing. Live SDK
subscription/authentication uses the SDK's networking behavior, while waiting
for the first trade uses the configured request timeout. Keep local clocks
synchronized. Provider availability and market hours are not guaranteed.
