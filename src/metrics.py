"""
Financial Metrics Module
========================
Calculates advanced financial metrics including Sharpe Ratio, Sortino Ratio,
Volatility, Max Drawdown, and daily/cumulative returns.

These metrics provide comprehensive risk-adjusted return analysis for
portfolio and individual security evaluation.

Author: Hermona Addisu
Date: November 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class FinancialMetrics:
    """
    Calculate comprehensive financial metrics for stock performance evaluation.
    
    Metrics computed:
    - Daily Return: Percentage change in closing price day-over-day
    - Cumulative Return: Total percentage gain from start to end
    - Volatility: Standard deviation of daily returns (annualized)
    - Sharpe Ratio: Risk-adjusted return (excess return per unit of risk)
    - Sortino Ratio: Downside risk-adjusted return (only penalizes negative volatility)
    - Max Drawdown: Largest peak-to-trough decline
    - Calmar Ratio: Return per unit of maximum drawdown
    - Win Rate: Percentage of positive daily returns
    
    Example:
        >>> metrics = FinancialMetrics(df, risk_free_rate=0.02)
        >>> sharpe = metrics.sharpe_ratio()
        >>> sortino = metrics.sortino_ratio()
        >>> summary = metrics.get_summary()
    """
    
    def __init__(self, df: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize Financial Metrics calculator.
        
        Args:
            df (pd.DataFrame): Stock data with 'Close' column and DatetimeIndex
            risk_free_rate (float): Annual risk-free rate for Sharpe/Sortino (default 2%)
        """
        self.df = df.copy()
        self.risk_free_rate = risk_free_rate
        
        # Ensure Close column exists
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column")
        
        # Calculate daily returns
        self.daily_returns = self.df['Close'].pct_change() * 100  # Percentage
        
    def daily_return(self) -> pd.Series:
        """
        Calculate daily percentage returns.
        
        Returns:
            pd.Series: Daily return percentages
        """
        return self.daily_returns
    
    def cumulative_return(self) -> float:
        """
        Calculate total cumulative return from start to end date.
        
        Formula: ((End Price - Start Price) / Start Price) * 100
        
        Returns:
            float: Cumulative return percentage
        """
        start_price = self.df['Close'].iloc[0]
        end_price = self.df['Close'].iloc[-1]
        return ((end_price - start_price) / start_price) * 100
    
    def volatility(self, annualized: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns).
        
        Args:
            annualized (bool): If True, annualize daily volatility (multiply by sqrt(252))
        
        Returns:
            float: Volatility as percentage
        """
        daily_vol = self.daily_returns.std()
        
        if annualized:
            return daily_vol * np.sqrt(252)  # 252 trading days per year
        return daily_vol
    
    def sharpe_ratio(self) -> float:
        """
        Calculate Sharpe Ratio (risk-adjusted return).
        
        Formula: (Mean Return - Risk-Free Rate) / Volatility
        
        Measures excess return per unit of total risk.
        Higher is better. Above 1.0 is good, above 2.0 is excellent.
        
        Returns:
            float: Sharpe ratio (dimensionless)
        """
        annual_return = self.daily_returns.mean() * 252  # Annualized
        annual_volatility = self.volatility(annualized=True)
        
        if annual_volatility == 0:
            return 0
        
        return (annual_return - self.risk_free_rate) / annual_volatility
    
    def sortino_ratio(self) -> float:
        """
        Calculate Sortino Ratio (downside risk-adjusted return).
        
        Similar to Sharpe but only penalizes downside volatility (negative returns).
        More favorable for investors than Sharpe because it ignores positive volatility.
        
        Formula: (Mean Return - Risk-Free Rate) / Downside Volatility
        
        Returns:
            float: Sortino ratio (dimensionless)
        """
        annual_return = self.daily_returns.mean() * 252
        
        # Downside returns (only negative returns)
        downside_returns = self.daily_returns[self.daily_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        
        if downside_volatility == 0:
            return 0
        
        return (annual_return - self.risk_free_rate) / downside_volatility
    
    def max_drawdown(self) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate maximum drawdown (largest peak-to-trough decline).
        
        Measures the worst-case loss from a peak to a subsequent trough.
        Important for risk assessment.
        
        Formula: (Trough Value - Peak Value) / Peak Value * 100
        
        Returns:
            Tuple[float, Timestamp, Timestamp]: (max_drawdown_pct, peak_date, trough_date)
        """
        cumulative = (1 + self.daily_returns / 100).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        max_dd = drawdown.min()
        trough_idx = drawdown.idxmin()
        
        # Find the peak that preceded this trough
        peak_idx = cumulative[:trough_idx].idxmax()
        
        return max_dd, peak_idx, trough_idx
    
    def calmar_ratio(self) -> float:
        """
        Calculate Calmar Ratio (return per unit of maximum drawdown).
        
        Formula: Annual Return / Absolute Max Drawdown
        
        Good ratio should be > 1.0. Useful for comparing return efficiency
        relative to worst-case loss.
        
        Returns:
            float: Calmar ratio (dimensionless)
        """
        annual_return = self.daily_returns.mean() * 252
        max_dd, _, _ = self.max_drawdown()
        
        if max_dd == 0:
            return 0
        
        return annual_return / abs(max_dd)
    
    def win_rate(self) -> float:
        """
        Calculate percentage of positive daily returns (win rate).
        
        Returns:
            float: Win rate as percentage (0-100)
        """
        wins = (self.daily_returns > 0).sum()
        total = len(self.daily_returns.dropna())
        
        if total == 0:
            return 0
        
        return (wins / total) * 100
    
    def best_day(self) -> Tuple[float, pd.Timestamp]:
        """
        Find best daily return.
        
        Returns:
            Tuple[float, Timestamp]: (best_return_pct, date)
        """
        best_return = self.daily_returns.max()
        best_date = self.daily_returns.idxmax()
        return best_return, best_date
    
    def worst_day(self) -> Tuple[float, pd.Timestamp]:
        """
        Find worst daily return.
        
        Returns:
            Tuple[float, Timestamp]: (worst_return_pct, date)
        """
        worst_return = self.daily_returns.min()
        worst_date = self.daily_returns.idxmin()
        return worst_return, worst_date
    
    def value_at_risk(self, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) - maximum expected loss at given confidence level.
        
        Args:
            confidence (float): Confidence level (0.90 = 90%, 0.95 = 95%, etc.)
        
        Returns:
            float: VaR as percentage (negative value, e.g., -5.2%)
        """
        return np.percentile(self.daily_returns.dropna(), (1 - confidence) * 100)
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get comprehensive summary of all metrics.
        
        Returns:
            Dict[str, float]: Dictionary with all calculated metrics
        """
        max_dd, peak_date, trough_date = self.max_drawdown()
        best_return, best_date = self.best_day()
        worst_return, worst_date = self.worst_day()
        
        return {
            'Cumulative Return (%)': self.cumulative_return(),
            'Annual Return (%)': self.daily_returns.mean() * 252,
            'Daily Volatility (%)': self.volatility(annualized=False),
            'Annual Volatility (%)': self.volatility(annualized=True),
            'Sharpe Ratio': self.sharpe_ratio(),
            'Sortino Ratio': self.sortino_ratio(),
            'Calmar Ratio': self.calmar_ratio(),
            'Max Drawdown (%)': max_dd,
            'Win Rate (%)': self.win_rate(),
            'Best Day (%)': best_return,
            'Worst Day (%)': worst_return,
            'VaR 95% (%)': self.value_at_risk(0.95),
        }
    
    def print_summary(self):
        """Print formatted summary of financial metrics."""
        summary = self.get_summary()
        
        print("=" * 70)
        print("FINANCIAL METRICS SUMMARY")
        print("=" * 70)
        
        print(f"\nRETURN METRICS:")
        print(f"  Cumulative Return:      {summary['Cumulative Return (%)']:>10.2f}%")
        print(f"  Annual Return:          {summary['Annual Return (%)']:>10.2f}%")
        
        print(f"\nVOLATILITY METRICS:")
        print(f"  Daily Volatility:       {summary['Daily Volatility (%)']:>10.2f}%")
        print(f"  Annual Volatility:      {summary['Annual Volatility (%)']:>10.2f}%")
        
        print(f"\nRISK-ADJUSTED RETURNS:")
        print(f"  Sharpe Ratio:           {summary['Sharpe Ratio']:>10.3f}")
        print(f"  Sortino Ratio:          {summary['Sortino Ratio']:>10.3f}")
        print(f"  Calmar Ratio:           {summary['Calmar Ratio']:>10.3f}")
        
        print(f"\nDRAWDOWN & EXTREMES:")
        print(f"  Max Drawdown:           {summary['Max Drawdown (%)']:>10.2f}%")
        print(f"  Win Rate:               {summary['Win Rate (%)']:>10.2f}%")
        print(f"  Best Day:               {summary['Best Day (%)']:>10.2f}%")
        print(f"  Worst Day:              {summary['Worst Day (%)']:>10.2f}%")
        
        print(f"\nRISK METRICS:")
        print(f"  Value at Risk (95%):    {summary['VaR 95% (%)']:>10.2f}%")
        print("=" * 70)


def calculate_metrics(df: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Convenience function to calculate all financial metrics at once.
    
    Args:
        df (pd.DataFrame): Stock data with 'Close' column
        risk_free_rate (float): Annual risk-free rate
    
    Returns:
        Dict[str, float]: All calculated metrics
    
    Example:
        >>> metrics = calculate_metrics(stock_df)
        >>> print(f"Sharpe Ratio: {metrics['Sharpe Ratio']}")
    """
    fm = FinancialMetrics(df, risk_free_rate)
    return fm.get_summary()
