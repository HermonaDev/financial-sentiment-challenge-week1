"""
Unit tests for sentiment analyzer module
"""

import unittest
import sys
sys.path.append('../')

from src.sentiment_analyzer import SentimentAnalyzer


class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for SentimentAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = SentimentAnalyzer()
    
    def test_positive_sentiment(self):
        """Test that positive text returns positive score"""
        text = "Stock hits all-time high with excellent earnings"
        score = self.analyzer.get_sentiment_score(text)
        self.assertGreater(score, 0, "Positive text should have positive score")
    
    def test_negative_sentiment(self):
        """Test that negative text returns negative score"""
        text = "Stock plummets amid terrible losses and layoffs"
        score = self.analyzer.get_sentiment_score(text)
        self.assertLess(score, 0, "Negative text should have negative score")
    
    def test_neutral_sentiment(self):
        """Test that neutral text returns score near zero"""
        text = "The company released a statement today"
        score = self.analyzer.get_sentiment_score(text)
        self.assertAlmostEqual(score, 0, delta=0.3, msg="Neutral text should have score near 0")
    
    def test_categorize_positive(self):
        """Test positive categorization"""
        category = self.analyzer.categorize_sentiment(0.5)
        self.assertEqual(category, 'Positive')
    
    def test_categorize_negative(self):
        """Test negative categorization"""
        category = self.analyzer.categorize_sentiment(-0.5)
        self.assertEqual(category, 'Negative')
    
    def test_categorize_neutral(self):
        """Test neutral categorization"""
        category = self.analyzer.categorize_sentiment(0.05)
        self.assertEqual(category, 'Neutral')


if __name__ == '__main__':
    unittest.main()