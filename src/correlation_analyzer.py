"""
Correlation Analysis Module
Calculates correlations between sentiment and stock returns
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional


class CorrelationAnalyzer:
    """
    Analyze correlations between news sentiment and stock price movements
    
    This class provides methods to:
    - Merge sentiment data with stock returns
    - Calculate Pearson correlation coefficients
    - Perform statistical significance testing
    - Analyze lagged correlations
    
    Attributes:
        min_data_points (int): Minimum data points required for correlation
    """
    
    def __init__(self, min_data_points: int = 30):
        """
        Initialize the CorrelationAnalyzer
        
        Args:
            min_data_points (int): Minimum overlapping data points required
        """
        self.min_data_points = min_data_points
    
    def merge_sentiment_and_returns(
        self, 
        sentiment_df: pd.DataFrame, 
        returns_df: pd.DataFrame,
        stock_symbol: str
    ) -> pd.DataFrame:
        """
        Merge sentiment data with stock returns for a specific stock
        
        Args:
            sentiment_df (pd.DataFrame): Daily sentiment data with columns 
                                        ['stock', 'date', 'avg_sentiment']
            returns_df (pd.DataFrame): Stock returns with columns 
                                      ['date', 'Daily_Return']
            stock_symbol (str): Stock ticker to analyze
            
        Returns:
            pd.DataFrame: Merged data with sentiment and returns aligned by date
        """
        # Filter sentiment for this stock
        stock_sentiment = sentiment_df[sentiment_df['stock'] == stock_symbol].copy()
        stock_sentiment['date'] = pd.to_datetime(stock_sentiment['date'])
        
        # Ensure returns date is datetime
        returns_df = returns_df.copy()
        returns_df['date'] = pd.to_datetime(returns_df['date'])
        
        # Merge on date
        merged = pd.merge(
            stock_sentiment[['date', 'avg_sentiment', 'article_count']], 
            returns_df[['date', 'Daily_Return']], 
            on='date', 
            how='inner'
        )
        
        # Remove NaN values
        merged = merged.dropna(subset=['avg_sentiment', 'Daily_Return'])
        
        return merged
    
    def calculate_correlation(
        self, 
        sentiment: pd.Series, 
        returns: pd.Series
    ) -> Tuple[float, float]:
        """
        Calculate Pearson correlation and p-value
        
        Args:
            sentiment (pd.Series): Sentiment scores
            returns (pd.Series): Stock returns
            
        Returns:
            tuple: (correlation_coefficient, p_value)
        """
        if len(sentiment) < self.min_data_points:
            return (np.nan, np.nan)
        
        # Calculate Pearson correlation
        corr_coef, p_value = stats.pearsonr(sentiment, returns)
        
        return (corr_coef, p_value)
    
    def analyze_stock_correlation(
        self,
        sentiment_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        stock_symbol: str
    ) -> Dict:
        """
        Complete correlation analysis for a single stock
        
        Args:
            sentiment_df (pd.DataFrame): Daily sentiment data
            returns_df (pd.DataFrame): Stock returns data
            stock_symbol (str): Stock ticker to analyze
            
        Returns:
            dict: Analysis results including correlation, p-value, sample size
        """
        # Merge data
        merged = self.merge_sentiment_and_returns(
            sentiment_df, returns_df, stock_symbol
        )
        
        if len(merged) < self.min_data_points:
            return {
                'stock': stock_symbol,
                'correlation': np.nan,
                'p_value': np.nan,
                'sample_size': len(merged),
                'significant': False,
                'status': 'insufficient_data'
            }
        
        # Calculate correlation
        corr, p_val = self.calculate_correlation(
            merged['avg_sentiment'], 
            merged['Daily_Return']
        )
        
        # Determine significance (p < 0.05)
        significant = p_val < 0.05 if not np.isnan(p_val) else False
        
        # Interpret strength
        if abs(corr) < 0.1:
            strength = 'Very Weak'
        elif abs(corr) < 0.3:
            strength = 'Weak'
        elif abs(corr) < 0.5:
            strength = 'Moderate'
        elif abs(corr) < 0.7:
            strength = 'Strong'
        else:
            strength = 'Very Strong'
        
        return {
            'stock': stock_symbol,
            'correlation': corr,
            'p_value': p_val,
            'sample_size': len(merged),
            'significant': significant,
            'strength': strength,
            'direction': 'Positive' if corr > 0 else 'Negative',
            'status': 'success'
        }
    
    def analyze_lagged_correlation(
        self,
        sentiment_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        stock_symbol: str,
        max_lag: int = 5
    ) -> pd.DataFrame:
        """
        Analyze correlation at different time lags
        
        Tests whether sentiment today correlates with returns in future days
        
        Args:
            sentiment_df (pd.DataFrame): Daily sentiment data
            returns_df (pd.DataFrame): Stock returns data
            stock_symbol (str): Stock ticker to analyze
            max_lag (int): Maximum number of days to lag
            
        Returns:
            pd.DataFrame: Correlations at each lag period
        """
        merged = self.merge_sentiment_and_returns(
            sentiment_df, returns_df, stock_symbol
        )
        
        if len(merged) < self.min_data_points:
            return pd.DataFrame()
        
        results = []
        
        for lag in range(0, max_lag + 1):
            # Shift returns forward by lag days
            merged[f'return_lag_{lag}'] = merged['Daily_Return'].shift(-lag)
            
            # Calculate correlation
            corr, p_val = self.calculate_correlation(
                merged['avg_sentiment'], 
                merged[f'return_lag_{lag}'].dropna()
            )
            
            results.append({
                'lag_days': lag,
                'correlation': corr,
                'p_value': p_val,
                'significant': p_val < 0.05 if not np.isnan(p_val) else False
            })
        
        return pd.DataFrame(results)
    
    def batch_analyze_correlations(
        self,
        sentiment_df: pd.DataFrame,
        stock_data_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Analyze correlations for multiple stocks at once
        
        Args:
            sentiment_df (pd.DataFrame): Daily sentiment data for all stocks
            stock_data_dict (dict): Dictionary mapping stock symbols to their
                                   price/return dataframes
            
        Returns:
            pd.DataFrame: Summary of correlations for all stocks
        """
        results = []
        
        for symbol, returns_df in stock_data_dict.items():
            result = self.analyze_stock_correlation(
                sentiment_df, returns_df, symbol
            )
            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        # Sort by absolute correlation value
        results_df['abs_correlation'] = results_df['correlation'].abs()
        results_df = results_df.sort_values('abs_correlation', ascending=False)
        results_df = results_df.drop('abs_correlation', axis=1)
        
        return results_df