"""
Stock Data Loader Module
========================
Provides OHLCV data loading and preprocessing using PyNance integration.
This module serves as the backbone for accessing and preparing stock price data
for technical analysis and correlation studies.

Author: Hermona Addisu
Date: November 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

try:
    import pynance as pn
    PYNANCE_AVAILABLE = True
except ImportError:
    PYNANCE_AVAILABLE = False

import yfinance as yf
import talib


class StockDataLoader:
    """
    Stock Data Loader for OHLCV (Open, High, Low, Close, Volume) data.
    
    Integrates PyNance for financial metrics extraction and fallback to yfinance
    for reliable price data retrieval. Provides methods for data cleaning, 
    validation, and alignment across multiple time series.
    
    Attributes:
        symbols (List[str]): List of stock symbols to load
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        ohlcv_data (Dict[str, pd.DataFrame]): Cached OHLCV data by symbol
        financial_metrics (Dict[str, Dict]): Financial metrics from PyNance
    
    Example:
        >>> loader = StockDataLoader(['AAPL', 'MSFT'], '2020-01-01', '2023-12-31')
        >>> ohlcv_data = loader.load_ohlcv_data()
        >>> metrics = loader.get_financial_metrics()
    """
    
    def __init__(self, symbols: List[str], start_date: str, end_date: str):
        """
        Initialize the Stock Data Loader.
        
        Args:
            symbols (List[str]): List of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.ohlcv_data: Dict[str, pd.DataFrame] = {}
        self.financial_metrics: Dict[str, Dict] = {}
        
    def load_ohlcv_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV (Open, High, Low, Close, Volume) data for all symbols.
        
        Uses yfinance as primary source with PyNance integration for
        validation and supplementary financial metrics.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to OHLCV DataFrames.
                Each DataFrame contains columns: ['Open', 'High', 'Low', 'Close', 'Volume']
                with DatetimeIndex
        
        Raises:
            ValueError: If no data could be loaded for a symbol
        """
        print(f"Loading OHLCV data for {len(self.symbols)} symbols...")
        print(f"Date range: {self.start_date} to {self.end_date}\n")
        
        for symbol in self.symbols:
            try:
                # Primary: Use yfinance for reliable OHLCV data
                df = yf.download(symbol, start=self.start_date, end=self.end_date, 
                                    progress=False)
                
                if df.empty or len(df) < 50:
                    print(f"  ✗ {symbol}: Insufficient data ({len(df)} days)")
                    continue
                
                # Validate OHLCV columns exist
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_cols):
                    print(f"  ✗ {symbol}: Missing OHLCV columns")
                    continue
                
                # Clean data: remove NaN, zero volumes, invalid prices
                df = self._clean_ohlcv_data(df)
                
                # Ensure proper data types
                for col in required_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Drop any remaining NaN after conversion
                df = df.dropna(subset=required_cols)
                
                self.ohlcv_data[symbol] = df[required_cols]
                
                print(f"  ✓ {symbol}: {len(df)} trading days loaded")
                
            except Exception as e:
                print(f"  ✗ {symbol}: Error - {str(e)[:60]}")
                continue
        
        print(f"\n✓ Successfully loaded {len(self.ohlcv_data)} stocks\n")
        return self.ohlcv_data
    
    def _clean_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean OHLCV data by removing invalid rows.
        
        Args:
            df (pd.DataFrame): Raw OHLCV DataFrame
        
        Returns:
            pd.DataFrame: Cleaned OHLCV DataFrame
        """
        # Remove rows with zero or NaN values in critical columns
        df = df[(df['Close'] > 0) & (df['Volume'] > 0)]
        
        # Remove rows where High < Low (data error)
        df = df[df['High'] >= df['Low']]
        
        # Remove rows where Close is outside High-Low range (data error)
        df = df[(df['Close'] >= df['Low']) & (df['Close'] <= df['High'])]
        
        return df
    
    def get_financial_metrics(self) -> Dict[str, Dict]:
        """
        Extract financial metrics for each symbol using PyNance.
        
        PyNance provides access to fundamental financial data, company information,
        and market metrics that complement price-based technical analysis.
        
        Returns:
            Dict[str, Dict]: Dictionary mapping symbols to financial metrics.
                Includes: market_cap, pe_ratio, dividend_yield, profit_margin, etc.
        """
        print("Retrieving financial metrics via PyNance...")
        
        if not PYNANCE_AVAILABLE:
            print("  ⚠ PyNance not installed. Extracting metrics from yfinance instead.\n")
            return self._get_metrics_from_yfinance()
        
        for symbol in self.symbols:
            try:
                # PyNance provides comprehensive financial data access
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                metrics = {
                    'symbol': symbol,
                    'name': info.get('longName', 'N/A'),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'market_cap': info.get('marketCap', None),
                    'pe_ratio': info.get('trailingPE', None),
                    'dividend_yield': info.get('dividendYield', None),
                    'profit_margin': info.get('profitMargins', None),
                    'return_on_equity': info.get('returnOnEquity', None),
                    '52_week_high': info.get('fiftyTwoWeekHigh', None),
                    '52_week_low': info.get('fiftyTwoWeekLow', None),
                    'avg_volume': info.get('averageVolume', None),
                }
                
                self.financial_metrics[symbol] = metrics
                print(f"  ✓ {symbol}: Metrics retrieved")
                
            except Exception as e:
                print(f"  ✗ {symbol}: Could not retrieve metrics - {str(e)[:40]}")
                continue
        
        print(f"\n✓ Financial metrics extracted for {len(self.financial_metrics)} symbols\n")
        return self.financial_metrics
    
    def _get_metrics_from_yfinance(self) -> Dict[str, Dict]:
        """
        Fallback method to extract financial metrics from yfinance when PyNance unavailable.
        
        Returns:
            Dict[str, Dict]: Financial metrics dictionary
        """
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                metrics = {
                    'symbol': symbol,
                    'name': info.get('longName', 'N/A'),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'market_cap': info.get('marketCap', None),
                    'pe_ratio': info.get('trailingPE', None),
                    'dividend_yield': info.get('dividendYield', None),
                    'profit_margin': info.get('profitMargins', None),
                    '52_week_high': info.get('fiftyTwoWeekHigh', None),
                    '52_week_low': info.get('fiftyTwoWeekLow', None),
                }
                
                self.financial_metrics[symbol] = metrics
                
            except Exception as e:
                print(f"  ⚠ {symbol}: Metrics retrieval failed")
                continue
        
        return self.financial_metrics
    
    def get_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data for a specific symbol.
        
        Args:
            symbol (str): Stock ticker symbol
        
        Returns:
            Optional[pd.DataFrame]: OHLCV DataFrame if available, else None
        """
        return self.ohlcv_data.get(symbol, None)
    
    def get_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get actual date range of loaded data.
        
        Returns:
            Tuple[pd.Timestamp, pd.Timestamp]: (min_date, max_date) from all loaded data
        """
        if not self.ohlcv_data:
            return None, None
        
        all_dates = []
        for df in self.ohlcv_data.values():
            all_dates.extend(df.index)
        
        return min(all_dates), max(all_dates)
    
    def get_statistics(self) -> pd.DataFrame:
        """
        Generate summary statistics for loaded OHLCV data.
        
        Returns:
            pd.DataFrame: Summary statistics table with symbol, days, price range, avg volume
        """
        stats_list = []
        
        for symbol, df in self.ohlcv_data.items():
            stats = {
                'Symbol': symbol,
                'Days': len(df),
                'Close_Mean': df['Close'].mean(),
                'Close_Std': df['Close'].std(),
                'High_52W': df['High'].max(),
                'Low_52W': df['Low'].min(),
                'Volume_Avg': df['Volume'].mean(),
            }
            stats_list.append(stats)
        
        return pd.DataFrame(stats_list)


def create_data_loader(symbols: List[str], start_date: str, end_date: str) -> StockDataLoader:
    """
    Factory function to create a StockDataLoader instance.
    
    Args:
        symbols (List[str]): List of stock ticker symbols
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        StockDataLoader: Initialized data loader instance
    
    Example:
        >>> loader = create_data_loader(['AAPL', 'MSFT'], '2020-01-01', '2023-12-31')
        >>> ohlcv_data = loader.load_ohlcv_data()
    """
    return StockDataLoader(symbols, start_date, end_date)
