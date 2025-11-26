"""
Technical Indicators Module
Calculates technical indicators for stock price data
"""

import pandas as pd
import numpy as np
from typing import Dict

# Optional TA-Lib integration: use TA-Lib functions when available for SMA/EMA/MACD/RSI
try:
    import talib
    HAS_TALIB = True
except Exception:
    talib = None
    HAS_TALIB = False


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
        if HAS_TALIB:
            # talib.SMA expects a numpy array
            return pd.Series(talib.SMA(df[column].values, timeperiod=window), index=df.index)
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
        if HAS_TALIB:
            # talib.EMA uses timeperiod argument
            return pd.Series(talib.EMA(df[column].values, timeperiod=span), index=df.index)
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
        if HAS_TALIB:
            macd, signal_line, histogram = talib.MACD(df[column].values,
                                                      fastperiod=fast,
                                                      slowperiod=slow,
                                                      signalperiod=signal)
            return {
                'MACD': pd.Series(macd, index=df.index),
                'MACD_Signal': pd.Series(signal_line, index=df.index),
                'MACD_Hist': pd.Series(histogram, index=df.index)
            }

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
        if HAS_TALIB:
            return pd.Series(talib.RSI(df[column].values, timeperiod=window), index=df.index)

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

    def validate_against_talib(self, df: pd.DataFrame, window: int = 14) -> Dict[str, Dict[str, float]]:
        """
        Validate pandas-based indicator calculations against TA-Lib (if available).

        Returns a dictionary of summary statistics (max_abs_diff, mean_abs_diff)
        for indicators: SMA_20, SMA_50, EMA_12, EMA_26, RSI, MACD components.
        """
        results = {}
        if not HAS_TALIB:
            return {"error": "TA-Lib not available in environment"}

        # Prepare a copy and compute both
        p = df.copy()
        # pandas-based
        pandas_sma20 = p['Close'].rolling(window=20).mean()
        pandas_sma50 = p['Close'].rolling(window=50).mean()
        pandas_ema12 = p['Close'].ewm(span=12, adjust=False).mean()
        pandas_ema26 = p['Close'].ewm(span=26, adjust=False).mean()
        pandas_macd = pandas_ema12 - pandas_ema26
        pandas_signal = pandas_macd.ewm(span=9, adjust=False).mean()
        pandas_hist = pandas_macd - pandas_signal
        pandas_rsi = TechnicalIndicators.calculate_rsi(p, window)

        # talib-based
        talib_sma20 = pd.Series(talib.SMA(p['Close'].values, timeperiod=20), index=p.index)
        talib_sma50 = pd.Series(talib.SMA(p['Close'].values, timeperiod=50), index=p.index)
        talib_ema12 = pd.Series(talib.EMA(p['Close'].values, timeperiod=12), index=p.index)
        talib_ema26 = pd.Series(talib.EMA(p['Close'].values, timeperiod=26), index=p.index)
        talib_macd, talib_signal, talib_hist = talib.MACD(p['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
        talib_macd = pd.Series(talib_macd, index=p.index)
        talib_signal = pd.Series(talib_signal, index=p.index)
        talib_hist = pd.Series(talib_hist, index=p.index)
        talib_rsi = pd.Series(talib.RSI(p['Close'].values, timeperiod=window), index=p.index)

        def summarize(a, b):
            diff = (a - b).abs().dropna()
            if diff.empty:
                return {"max_abs_diff": float('nan'), "mean_abs_diff": float('nan')}
            return {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}

        results['SMA_20'] = summarize(pandas_sma20, talib_sma20)
        results['SMA_50'] = summarize(pandas_sma50, talib_sma50)
        results['EMA_12'] = summarize(pandas_ema12, talib_ema12)
        results['EMA_26'] = summarize(pandas_ema26, talib_ema26)
        results['MACD'] = summarize(pandas_macd, talib_macd)
        results['MACD_Signal'] = summarize(pandas_signal, talib_signal)
        results['MACD_Hist'] = summarize(pandas_hist, talib_hist)
        results['RSI'] = summarize(pandas_rsi, talib_rsi)

        return results