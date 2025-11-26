"""
Run TA-Lib vs pandas indicator validation for a sample stock and save results.
Usage: python scripts/run_talib_validation.py
"""
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
data_dir = repo_root / 'data' / 'yfinance_data' / 'Data'
out_dir = repo_root / 'reports' / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)

# locate a sample CSV
csv_files = list(data_dir.glob('*.csv'))
if not csv_files:
    print('No stock CSV files found under', data_dir)
    sys.exit(1)

sample = csv_files[0]
symbol = sample.stem
print(f'Using sample file: {sample.name}')

import pandas as pd
import importlib.util
from pathlib import Path as _P

# Load `src/technical_indicators.py` directly to avoid importing package-level dependencies
ti_path = repo_root / 'src' / 'technical_indicators.py'
spec = importlib.util.spec_from_file_location('technical_indicators_local', str(ti_path))
ti_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ti_mod)
TechnicalIndicators = ti_mod.TechnicalIndicators

# load sample
df = pd.read_csv(sample, parse_dates=['Date']).sort_values('Date').set_index('Date')

ti = TechnicalIndicators()
validation = ti.validate_against_talib(df)

out_file = out_dir / f'validation_{symbol}.json'
with out_file.open('w') as f:
    json.dump(validation, f, indent=2)

print('Validation results saved to', out_file)
print(json.dumps(validation, indent=2))
