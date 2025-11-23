"""
Data Loading and Preprocessing Module
Handles loading and initial preprocessing of financial news data
"""

import pandas as pd
from datetime import datetime
from typing import Optional, Tuple


class NewsDataLoader:
    """
    Class for loading and preprocessing financial news data
    
    Attributes:
        data_path (str): Path to the CSV file
        df (pd.DataFrame): Loaded dataframe
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the NewsDataLoader
        
        Args:
            data_path (str): Path to the raw data CSV file
        """
        self.data_path = data_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the news data from CSV
        
        Returns:
            pd.DataFrame: Loaded news data
        """
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Loaded {len(self.df):,} records from {self.data_path}")
            return self.df
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def preprocess_dates(self) -> pd.DataFrame:
        """
        Parse and preprocess date columns
        
        Returns:
            pd.DataFrame: Data with processed date columns
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Convert to datetime
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        self.df['date_only'] = self.df['date'].dt.date
        
        # Extract date components
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day'] = self.df['date'].dt.day
        self.df['hour'] = self.df['date'].dt.hour
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        
        print(f"✓ Dates preprocessed: {self.df['date'].min()} to {self.df['date'].max()}")
        return self.df
    
    def get_basic_stats(self) -> dict:
        """
        Get basic statistics about the dataset
        
        Returns:
            dict: Dictionary containing basic statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        stats = {
            'total_articles': len(self.df),
            'unique_stocks': self.df['stock'].nunique(),
            'unique_publishers': self.df['publisher'].nunique(),
            'date_range': (self.df['date'].min(), self.df['date'].max()),
            'missing_values': self.df.isnull().sum().to_dict()
        }
        return stats