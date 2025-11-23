"""
Financial Sentiment Analysis Package
=====================================

A modular toolkit for analyzing financial news sentiment and correlating
it with stock price movements.

Modules:
--------
- data_loader: Load and preprocess financial news data
- sentiment_analyzer: Perform sentiment analysis on headlines
- technical_indicators: Calculate stock technical indicators
- correlation_analyzer: Correlate sentiment with price movements

Quick Start:
-----------
>>> from src import FinancialAnalysisPipeline
>>> pipeline = FinancialAnalysisPipeline('data/news.csv')
>>> results = pipeline.run_full_analysis()

Individual Components:
---------------------
>>> from src import NewsDataLoader, SentimentAnalyzer, TechnicalIndicators
>>> 
>>> # Load data
>>> loader = NewsDataLoader('data/news.csv')
>>> news_df = loader.load_data()
>>> 
>>> # Analyze sentiment
>>> analyzer = SentimentAnalyzer()
>>> news_df = analyzer.analyze_dataframe(news_df)
>>> 
>>> # Calculate indicators
>>> indicators = TechnicalIndicators()
>>> stock_df = indicators.calculate_all_indicators(stock_df)

Author: Hermona (UC Berkeley CS)
Version: 0.1.0
License: MIT
"""

from .data_loader import NewsDataLoader
from .sentiment_analyzer import SentimentAnalyzer
from .technical_indicators import TechnicalIndicators
from .correlation_analyzer import CorrelationAnalyzer

# Version info
__version__ = '0.1.0'
__author__ = 'Hermona'
__email__ = 'hermona@example.com'

# Public API
__all__ = [
    'NewsDataLoader',
    'SentimentAnalyzer', 
    'TechnicalIndicators',
    'CorrelationAnalyzer', 
]

# Package-level convenience function
def get_version():
    """Return the package version string"""
    return __version__


def quick_sentiment_analysis(data_path: str) -> dict:
    """
    Convenience function for quick sentiment analysis
    
    Args:
        data_path (str): Path to news CSV file
        
    Returns:
        dict: Summary of sentiment analysis results
        
    Example:
        >>> results = quick_sentiment_analysis('data/news.csv')
        >>> print(results['mean_sentiment'])
    """
    loader = NewsDataLoader(data_path)
    news_df = loader.load_data()
    news_df = loader.preprocess_dates()
    
    analyzer = SentimentAnalyzer()
    news_df = analyzer.analyze_dataframe(news_df)
    summary = analyzer.get_sentiment_summary(news_df)
    
    return summary