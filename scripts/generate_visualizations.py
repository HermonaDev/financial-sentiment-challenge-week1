"""
Generate report visualizations and save PNGs to `visualizations/`.

Notes:
- Uses NLTK VADER for sentiment (no TextBlob required).
- For performance, the script processes the news CSV in chunks and aggregates daily sentiment.
- Outputs (saved in `visualizations/`):
  - sentiment_analysis_44k.png
  - sentiment_categories.png
  - articles_over_time.png
  - top_stocks_coverage.png
  - technical_indicators_example.png
  - stock_metrics_comparison.png
  - correlation_summary.png

Run: python scripts/generate_visualizations.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
from pathlib import Path

sns.set_style('whitegrid')
VIS_DIR = Path('visualizations')
VIS_DIR.mkdir(exist_ok=True)

NEWS_CSV = Path('data/newsData/raw_analyst_ratings.csv')
STOCK_DIR = Path('data/yfinance_data/Data')
CORR_CSV = Path('notebooks/correlation_results.csv')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Helper: compute compound score
def compound_score(text):
    try:
        return sia.polarity_scores(str(text))['compound']
    except Exception:
        return np.nan

# 1) Process news headlines in chunks to compute sentiment and basic aggregates
print('Processing news data (chunked)...')
chunksize = 100000
sentiment_chunks = []
stock_counts = {}
date_counts = {}

for chunk in pd.read_csv(NEWS_CSV, chunksize=chunksize, usecols=['headline','date','stock']):
    # parse date only
    chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
    chunk = chunk.dropna(subset=['date'])
    chunk['date_only'] = chunk['date'].dt.date

    # compute compound sentiment for headlines in this chunk
    chunk['compound'] = chunk['headline'].apply(compound_score)

    # aggregate daily mean sentiment across all stocks
    daily = chunk.groupby('date_only')['compound'].agg(['mean','count']).reset_index()
    daily.columns = ['date_only','daily_sent_mean','daily_count']
    sentiment_chunks.append(daily)

    # update stock counts
    vc = chunk['stock'].value_counts()
    for s,c in vc.items():
        stock_counts[s] = stock_counts.get(s,0) + int(c)

    # update date counts
    dc = daily.set_index('date_only')['daily_count'].to_dict()
    for d,c in dc.items():
        date_counts[d] = date_counts.get(d,0) + int(c)

# Concatenate daily aggregates
sentiment_daily = pd.concat(sentiment_chunks).groupby('date_only').agg({'daily_sent_mean':'mean','daily_count':'sum'}).reset_index()
sentiment_daily['date_only'] = pd.to_datetime(sentiment_daily['date_only'])
print('News processing done. Days processed:', len(sentiment_daily))

# Overall sentiment distribution (sample headline-level distribution for plotting)
print('Sampling headlines for histogram (for performance)')
sample_df = pd.read_csv(NEWS_CSV, usecols=['headline'], nrows=200000)
sample_df['compound'] = sample_df['headline'].apply(compound_score)

# Categorize sentiment for sampled headlines
def categorize(compound):
    if pd.isna(compound):
        return 'Unknown'
    if compound > 0.1:
        return 'Positive'
    if compound < -0.1:
        return 'Negative'
    return 'Neutral'

sample_df['category'] = sample_df['compound'].apply(categorize)
cat_counts = sample_df['category'].value_counts()

# Top stocks coverage
stock_counts_series = pd.Series(stock_counts).sort_values(ascending=False)
top_stocks = stock_counts_series.head(15)

# --- Create visualizations ---
print('Creating visualization: sentiment_analysis_44k.png')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram of compound scores (sample)
axes[0].hist(sample_df['compound'].dropna(), bins=60, color='#4C72B0', edgecolor='k', alpha=0.8)
mean_comp = sample_df['compound'].mean()
axes[0].axvline(mean_comp, color='red', linestyle='--', label=f"Mean: {mean_comp:.2f}")
axes[0].set_title('Headline Sentiment (sample of headlines)')
axes[0].set_xlabel('Compound Sentiment Score')
axes[0].set_ylabel('Count')
axes[0].legend()

# Pie chart of categories
axes[1].pie(cat_counts.values, labels=cat_counts.index, autopct='%1.1f%%', startangle=140, colors=['#2ca02c','#1f77b4','#d62728'])
axes[1].set_title('Sentiment Categories (sample)')

# Rolling mean over time
sentiment_daily_sorted = sentiment_daily.sort_values('date_only').set_index('date_only')
rolling_30 = sentiment_daily_sorted['daily_sent_mean'].rolling(window=30, min_periods=5).mean()
axes[2].plot(sentiment_daily_sorted.index, sentiment_daily_sorted['daily_sent_mean'], alpha=0.4, label='Daily mean')
axes[2].plot(rolling_30.index, rolling_30.values, color='red', linewidth=2, label='30-day rolling mean')
axes[2].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[2].set_title('Daily Average Sentiment (rolling)')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Avg Sentiment')
axes[2].legend()

plt.tight_layout()
plt.savefig(VIS_DIR / 'sentiment_analysis_44k.png', dpi=300, bbox_inches='tight')
plt.close()

# Sentiment categories bar chart
print('Creating visualization: sentiment_categories.png')
plt.figure(figsize=(6,4))
sns.barplot(x=cat_counts.index, y=cat_counts.values, palette=['#2ca02c','#1f77b4','#d62728'])
plt.title('Headline Sentiment Categories (sample)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(VIS_DIR / 'sentiment_categories.png', dpi=300, bbox_inches='tight')
plt.close()

# Articles over time
print('Creating visualization: articles_over_time.png')
plt.figure(figsize=(12,4))
monthly = sentiment_daily_sorted['daily_count'].resample('M').sum()
monthly.plot()
plt.title('Articles Published Per Month')
plt.xlabel('Month')
plt.ylabel('Article Count')
plt.tight_layout()
plt.savefig(VIS_DIR / 'articles_over_time.png', dpi=300, bbox_inches='tight')
plt.close()

# Top stocks coverage
print('Creating visualization: top_stocks_coverage.png')
plt.figure(figsize=(8,6))
sns.barplot(y=top_stocks.index, x=top_stocks.values, palette='Blues_r')
plt.title('Top 15 Stocks by News Coverage (sampled)')
plt.xlabel('Article Count')
plt.ylabel('Stock')
plt.tight_layout()
plt.savefig(VIS_DIR / 'top_stocks_coverage.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Technical indicators example (use AAPL as sample) ---
print('Creating technical indicators visualization (AAPL)')
# Load AAPL
aapl = pd.read_csv(STOCK_DIR / 'AAPL.csv', parse_dates=['Date'])
aapl = aapl.sort_values('Date').set_index('Date')
# Simple moving averages
aapl['SMA20'] = aapl['Close'].rolling(window=20).mean()
aapl['SMA50'] = aapl['Close'].rolling(window=50).mean()
# RSI14
delta = aapl['Close'].diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
ema_up = up.ewm(com=13, adjust=False).mean()
ema_down = down.ewm(com=13, adjust=False).mean()
rs = ema_up / ema_down
aapl['RSI14'] = 100 - (100 / (1 + rs))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8), gridspec_kw={'height_ratios':[3,1]})
ax1.plot(aapl.index, aapl['Close'], label='Close', color='#1f77b4')
ax1.plot(aapl.index, aapl['SMA20'], label='SMA20', color='#ff7f0e')
ax1.plot(aapl.index, aapl['SMA50'], label='SMA50', color='#2ca02c')
ax1.set_title('AAPL: Price with SMA20/50')
ax1.legend()

ax2.plot(aapl.index, aapl['RSI14'], label='RSI14', color='#9467bd')
ax2.axhline(70, color='red', linestyle='--')
ax2.axhline(30, color='green', linestyle='--')
ax2.set_title('AAPL: RSI (14)')
ax2.set_ylim(0,100)
ax2.legend()

plt.tight_layout()
plt.savefig(VIS_DIR / 'technical_indicators_example.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Stock metrics comparison ---
print('Creating stock metrics comparison')
stocks = []
metrics = []
for f in STOCK_DIR.glob('*.csv'):
    sym = f.stem
    df = pd.read_csv(f, parse_dates=['Date']).sort_values('Date').set_index('Date')
    df = df.dropna(subset=['Close'])
    # daily returns
    df['ret'] = df['Close'].pct_change()
    ann_return = (1 + df['ret'].mean()) ** 252 - 1
    ann_vol = df['ret'].std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    metrics.append({'stock': sym, 'ann_return': ann_return*100, 'ann_vol': ann_vol*100, 'sharpe': sharpe})

metrics_df = pd.DataFrame(metrics).set_index('stock').sort_values('ann_return', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14,5))
metrics_df['ann_return'].plot(kind='bar', ax=axes[0], color='tab:green')
axes[0].set_title('Annualized Return (%)')
axes[0].set_ylabel('%')

metrics_df['ann_vol'].plot(kind='bar', ax=axes[1], color='tab:red')
axes[1].set_title('Annualized Volatility (%)')
axes[1].set_ylabel('%')

plt.tight_layout()
plt.savefig(VIS_DIR / 'stock_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Correlation summary chart using correlation_results.csv ---
if CORR_CSV.exists():
    print('Creating correlation summary chart from correlation_results.csv')
    corr_df = pd.read_csv(CORR_CSV)
    # drop code fences if present
    if corr_df.columns.tolist()[0].startswith('Stock') and corr_df.shape[1] >= 3:
        pass
    # Plot
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
    colors = ['#2ca02c' if p<0.05 else '#d62728' for p in corr_df['P_Value']]
    ax1.barh(corr_df['Stock'], corr_df['Correlation'], color=colors)
    ax1.set_xlabel('Pearson Correlation Coefficient')
    ax1.set_title('Sentiment-Return Correlation by Stock')
    ax1.axvline(0, color='black', linewidth=0.7)

    ax2.bar(corr_df['Stock'], corr_df['P_Value'], color=colors)
    ax2.axhline(0.05, color='black', linestyle='--', label='α = 0.05')
    ax2.set_ylabel('P-value')
    ax2.set_title('P-values (Significance)')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'correlation_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
else:
    print('Correlation results CSV not found; skipping correlation_summary.png')

print('All visualizations created (available ones).')

