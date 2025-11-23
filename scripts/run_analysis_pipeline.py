"""
Integrated Analysis Pipeline
Runs the complete sentiment-to-price analysis workflow
"""

import sys
sys.path.append('../')

from src.data_loader import NewsDataLoader
from src.sentiment_analyzer import SentimentAnalyzer
from src.technical_indicators import TechnicalIndicators
import pandas as pd
import yfinance as yf


def run_full_pipeline(data_path: str, top_n_stocks: int = 5):
    """
    Execute the complete analysis pipeline
    
    Args:
        data_path (str): Path to news data CSV
        top_n_stocks (int): Number of top stocks to analyze
    """
    print("="*60)
    print("FINANCIAL SENTIMENT ANALYSIS PIPELINE")
    print("="*60)
    
    # Step 1: Load and preprocess news data
    print("\n[1/5] Loading news data...")
    loader = NewsDataLoader(data_path)
    news_df = loader.load_data()
    news_df = loader.preprocess_dates()
    
    # Step 2: Perform sentiment analysis
    print("\n[2/5] Analyzing sentiment...")
    analyzer = SentimentAnalyzer()
    news_df = analyzer.analyze_dataframe(news_df)
    sentiment_summary = analyzer.get_sentiment_summary(news_df)
    print(f"  Mean sentiment: {sentiment_summary['mean']:.4f}")
    print(f"  Distribution: {sentiment_summary['distribution']}")
    
    # Step 3: Aggregate daily sentiment
    print("\n[3/5] Aggregating daily sentiment...")
    daily_sentiment = analyzer.aggregate_daily_sentiment(news_df)
    
    # Step 4: Download and analyze stock data
    print(f"\n[4/5] Downloading stock data for top {top_n_stocks} stocks...")
    top_stocks = news_df['stock'].value_counts().head(top_n_stocks).index.tolist()
    
    tech_indicators = TechnicalIndicators()
    stock_data = {}
    
    for symbol in top_stocks:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start='2020-01-01', end='2024-12-31')
            if len(df) > 100:
                df = tech_indicators.calculate_all_indicators(df)
                stock_data[symbol] = df
                print(f"  ✓ {symbol}: {len(df)} days analyzed")
        except Exception as e:
            print(f"  ✗ {symbol}: {str(e)[:30]}")
    
    # Step 5: Merge and correlate
    print("\n[5/5] Calculating correlations...")
    correlations = {}
    
    for symbol in stock_data.keys():
        # Merge sentiment with returns
        stock_sent = daily_sentiment[daily_sentiment['stock'] == symbol].copy()
        stock_sent['date'] = pd.to_datetime(stock_sent['date'])
        
        returns_df = stock_data[symbol].reset_index()
        returns_df['date'] = pd.to_datetime(returns_df['Date'].dt.date)
        
        merged = pd.merge(stock_sent, returns_df[['date', 'Daily_Return']], 
                         on='date', how='inner')
        
        if len(merged) > 30:
            corr = merged['avg_sentiment'].corr(merged['Daily_Return'])
            correlations[symbol] = corr
            print(f"  {symbol}: r = {corr:.4f}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    
    return {
        'news_df': news_df,
        'daily_sentiment': daily_sentiment,
        'stock_data': stock_data,
        'correlations': correlations
    }


if __name__ == "__main__":
    # Example usage
    results = run_full_pipeline('../data/raw_analyst_ratings.csv', top_n_stocks=5)
    print(f"\n✓ Analysis complete for {len(results['correlations'])} stocks")