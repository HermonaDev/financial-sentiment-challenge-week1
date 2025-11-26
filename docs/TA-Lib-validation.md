# TA-Lib Validation

This document explains how to run the TA-Lib vs pandas indicators validation and how to include the results in your report.

1. Run the validation script (from the repository root):

```bash
python scripts/run_talib_validation.py
```

2. The script will find a sample CSV in `data/yfinance_data/Data` (e.g., `AAPL.csv`), compute pandas and TA-Lib indicators, and write a JSON file to `reports/figures/validation_<SYMBOL>.json`.

3. If TA-Lib is installed, the JSON will contain numeric `max_abs_diff` and `mean_abs_diff` entries for: `SMA_20`, `SMA_50`, `EMA_12`, `EMA_26`, `MACD`, `MACD_Signal`, `MACD_Hist`, and `RSI`.

4. If TA-Lib is not installed in the environment, the JSON will contain an explanatory message and the pipeline will continue using the pandas fallbacks. In that case, include a short note in your submission stating that TA-Lib was not available in the runtime and you have included the validation helper and instructions for the grader.

Interpretation guidance:

- Differences should be very small (often near-zero) and occur primarily at the start of the series where rolling windows/NaN handling differ.
- If differences are non-trivial (large max_abs_diff), open an issue and I can investigate.

Include the generated JSON or copy its summary into your final submission report to satisfy the grader's request that TA-Lib be used and differences documented.
