"""
Technical Indicators Module
Calculates technical indicators for stock price data
"""

import pandas as pd
import numpy as np
from typing import Dict


class TechnicalIndicators:
    """
    Class for calculating technical indicators on stock price data
    
    Methods calculate various indicators: SMA, EMA, MACD, RSI, Bollinger Bands
    """
    
    @staticmethod
    def calculate_sma(df: pd.DataFrame, column: str = 'Close', window: int = 20) -> pd.Series:
        """
        Calculate Simple Moving Average
        
        Args:
            df (pd.DataFrame): Stock price dataframe
            column (str): Column name to calculate SMA on
            window (int): Rolling window size
            
        Returns:
            pd.Series: SMA values
        """
        return df[column].rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(df: pd.DataFrame, column: str = 'Close', span: int = 12) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Args:
            df (pd.DataFrame): Stock price dataframe
            column (str): Column name to calculate EMA on
            span (int): Span for exponential weighting
            
        Returns:
            pd.Series: EMA values
        """
        return df[column].ewm(span=span, adjust=False).mean()
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame, column: str = 'Close', 
                      fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            df (pd.DataFrame): Stock price dataframe
            column (str): Column name to calculate MACD on
            fast (int): Fast EMA period
            slow (int): Slow EMA period
            signal (int): Signal line period
            
        Returns:
            dict: Dictionary with MACD, Signal, and Histogram
        """
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return {
            'MACD': macd,
            'MACD_Signal': signal_line,
            'MACD_Hist': histogram
        }
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, column: str = 'Close', window: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index)
        
        Args:
            df (pd.DataFrame): Stock price dataframe
            column (str): Column name to calculate RSI on
            window (int): RSI period
            
        Returns:
            pd.Series: RSI values (0-100)
        """
        delta = df[column].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, column: str = 'Close', 
                                 window: int = 20, num_std: int = 2) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            df (pd.DataFrame): Stock price dataframe
            column (str): Column name to calculate bands on
            window (int): Rolling window size
            num_std (int): Number of standard deviations
            
        Returns:
            dict: Dictionary with Upper, Middle, and Lower bands
        """
        middle_band = df[column].rolling(window=window).mean()
        std = df[column].rolling(window=window).std()
        
        upper_band = middle_band + (num_std * std)
        lower_band = middle_band - (num_std * std)
        
        return {
            'BB_Upper': upper_band,
            'BB_Middle': middle_band,
            'BB_Lower': lower_band
        }
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators at once
        
        Args:
            df (pd.DataFrame): Stock price dataframe with OHLCV data
            
        Returns:
            pd.DataFrame: Dataframe with all indicators added
        """
        data = df.copy()
        
        # Moving Averages
        data['SMA_20'] = self.calculate_sma(data, window=20)
        data['SMA_50'] = self.calculate_sma(data, window=50)
        data['EMA_12'] = self.calculate_ema(data, span=12)
        data['EMA_26'] = self.calculate_ema(data, span=26)
        
        # MACD
        macd_data = self.calculate_macd(data)
        data['MACD'] = macd_data['MACD']
        data['MACD_Signal'] = macd_data['MACD_Signal']
        data['MACD_Hist'] = macd_data['MACD_Hist']
        
        # RSI
        data['RSI'] = self.calculate_rsi(data)
        
        # Bollinger Bands
        bb_data = self.calculate_bollinger_bands(data)
        data['BB_Upper'] = bb_data['BB_Upper']
        data['BB_Middle'] = bb_data['BB_Middle']
        data['BB_Lower'] = bb_data['BB_Lower']
        
        # Daily Returns
        data['Daily_Return'] = data['Close'].pct_change() * 100
        
        # Volatility
        data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
        
        print("✓ All technical indicators calculated")
        return data