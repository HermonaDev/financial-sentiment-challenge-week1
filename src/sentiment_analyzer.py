"""
Sentiment Analysis Module
Performs sentiment analysis on financial news headlines using TextBlob
"""

import pandas as pd
from textblob import TextBlob
from typing import List, Dict


class SentimentAnalyzer:
    """
    Class for performing sentiment analysis on news headlines
    
    Attributes:
        positive_threshold (float): Threshold for positive sentiment
        negative_threshold (float): Threshold for negative sentiment
    """
    
    def __init__(self, positive_threshold: float = 0.1, negative_threshold: float = -0.1):
        """
        Initialize the SentimentAnalyzer
        
        Args:
            positive_threshold (float): Minimum score for positive sentiment
            negative_threshold (float): Maximum score for negative sentiment
        """
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
    
    def get_sentiment_score(self, text: str) -> float:
        """
        Calculate sentiment polarity score for a text
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            float: Sentiment score between -1 (negative) and +1 (positive)
        """
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity
        except Exception as e:
            print(f"Error analyzing text: {str(e)[:50]}")
            return 0.0
    
    def categorize_sentiment(self, score: float) -> str:
        """
        Categorize sentiment score into Positive/Neutral/Negative
        
        Args:
            score (float): Sentiment polarity score
            
        Returns:
            str: Sentiment category
        """
        if score > self.positive_threshold:
            return 'Positive'
        elif score < self.negative_threshold:
            return 'Negative'
        else:
            return 'Neutral'
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'headline') -> pd.DataFrame:
        """
        Analyze sentiment for all texts in a dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of column containing text to analyze
            
        Returns:
            pd.DataFrame: Dataframe with added sentiment columns
        """
        print(f"Analyzing sentiment for {len(df):,} records...")
        
        # Calculate sentiment scores
        df['sentiment'] = df[text_column].apply(self.get_sentiment_score)
        df['sentiment_category'] = df['sentiment'].apply(self.categorize_sentiment)
        
        print("✓ Sentiment analysis complete")
        return df
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of sentiment analysis
        
        Args:
            df (pd.DataFrame): Dataframe with sentiment column
            
        Returns:
            dict: Summary statistics
        """
        if 'sentiment' not in df.columns:
            raise ValueError("Sentiment column not found. Run analyze_dataframe() first.")
        
        summary = {
            'mean': df['sentiment'].mean(),
            'median': df['sentiment'].median(),
            'std': df['sentiment'].std(),
            'min': df['sentiment'].min(),
            'max': df['sentiment'].max(),
            'distribution': df['sentiment_category'].value_counts().to_dict()
        }
        return summary
    
    def aggregate_daily_sentiment(self, df: pd.DataFrame, date_column: str = 'date_only') -> pd.DataFrame:
        """
        Aggregate sentiment scores by stock and date
        
        Args:
            df (pd.DataFrame): Input dataframe with sentiment scores
            date_column (str): Name of date column
            
        Returns:
            pd.DataFrame: Aggregated daily sentiment per stock
        """
        daily_sentiment = df.groupby(['stock', date_column]).agg({
            'sentiment': ['mean', 'std', 'count']
        }).reset_index()
        
        daily_sentiment.columns = ['stock', 'date', 'avg_sentiment', 'sentiment_std', 'article_count']
        
        print(f"✓ Aggregated to {len(daily_sentiment):,} daily records")
        return daily_sentiment