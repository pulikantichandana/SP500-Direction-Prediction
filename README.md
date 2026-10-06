# S&P 500 Direction Prediction

A deep learning pipeline that predicts the next-day direction (up / flat / down) of the S&P 500 index, combining a Transformer with attention pooling, an LSTM path, a stacked ensemble, and news/sentiment signals.

## What it does

- **Data ingestion**: Yahoo Finance (`^GSPC` with `SPY` fallback), plus VIX, 10Y Treasury yield (TNX), Gold, and constituent stocks as auxiliary indicators.
- **Feature engineering**: returns, rolling volatility, moving-average deltas, RSI, volume ratio, high-low range.
- **Sentiment**: headlines scored with FinBERT and blended in as an extra feature (via `headlines.json` or the Reddit/Gemini APIs).
- **Modeling**: Transformer (attention pooling) + LSTM path, stacked with feature-based learners into a meta-model.
- **Tri-class labeling** (up/flat/down) with abstention on low-magnitude/ambiguous days.
- **Conformal prediction** for calibrated confidence bands, and **VIX-based regime gating** to separate low/high volatility regimes.
- **Walk-forward cross-validation** to avoid look-ahead leakage during hyperparameter selection.

See [final.md](final.md) for the full write-up: measured/expected accuracy ranges, why daily direction accuracy realistically tops out well below 80%, a step-by-step pipeline walkthrough, sentiment limitations, and prioritized next steps.

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your own API keys:

```bash
cp .env.example .env
```

```
FRED_API_KEY=
REDDIT_CLIENT_SECRET=
GEMINI_API_KEY=
REDDIT_CLIENT_ID=
```

- `FRED_API_KEY` — [fredaccount.stlouisfed.org/apikeys](https://fredaccount.stlouisfed.org/apikeys)
- `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` — [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
- `GEMINI_API_KEY` — [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

`.env` is gitignored and loaded automatically at runtime; if a key is missing, the script prints a warning rather than failing silently.

## Run

```bash
python3 setup_and_run.py
```

This installs any missing dependencies, then trains and evaluates the ensemble end to end.

## Realistic expectations

Daily S&P 500 direction prediction is dominated by noise — published results on broad indices sit in the ~52–58% range for well-constructed models. This pipeline's tri-class + abstention + regime gating setup aims for higher *selective* accuracy (roughly 65–70%) on the highest-confidence subset of predictions, at the cost of trading less often. Details and caveats are in [final.md](final.md).

> **Note:** This was built as a coursework/academic project to explore deep learning and ensemble methods on financial time series. It is not intended for real trading or investment use, and nothing here is financial advice.
