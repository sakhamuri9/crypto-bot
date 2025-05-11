"""
Sentiment Analysis Integration for Trading Strategies

This module implements sentiment analysis from news and social media data
to enhance trading signals with market sentiment information.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import logging
import json
import requests
from textblob import TextBlob
import re
from bs4 import BeautifulSoup
import time
import random
import tweepy
from tweepy import OAuthHandler
import configparser
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sentiment_analysis.log')
    ]
)

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Sentiment analyzer for cryptocurrency news and social media data.
    """
    
    def __init__(self, symbols, lookback_days=7):
        """
        Initialize the sentiment analyzer.
        
        Args:
            symbols (list): List of symbols to analyze
            lookback_days (int): Number of days to look back
        """
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.news_sources = [
            'https://cryptonews.com/',
            'https://cointelegraph.com/',
            'https://www.coindesk.com/'
        ]
        self.social_sources = [
            'twitter',
            'reddit'
        ]
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 11.5; rv:90.0) Gecko/20100101 Firefox/90.0'
        ]
        
        # Initialize Twitter API client
        self.twitter_api = None
        self.twitter_client_v2 = None
        self.setup_twitter_api()
    
    def setup_twitter_api(self):
        """
        Set up Twitter API client using credentials from config file.
        """
        try:
            config = configparser.ConfigParser()
            config.read('twitter_config.ini')
            
            consumer_key = config['twitter']['consumer_key']
            consumer_secret = config['twitter']['consumer_secret']
            access_token = config['twitter']['access_token']
            access_token_secret = config['twitter']['access_token_secret']
            bearer_token = config['twitter']['bearer_token']
            
            if consumer_key == 'YOUR_CONSUMER_KEY':
                logger.warning("Twitter API credentials not configured. Using simulated data.")
                return
            
            auth = OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_token, access_token_secret)
            self.twitter_api = tweepy.API(auth, wait_on_rate_limit=True)
            
            self.twitter_client_v2 = tweepy.Client(
                bearer_token=bearer_token,
                consumer_key=consumer_key,
                consumer_secret=consumer_secret,
                access_token=access_token,
                access_token_secret=access_token_secret,
                wait_on_rate_limit=True
            )
            
            logger.info("Twitter API client initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up Twitter API client: {e}")
            logger.warning("Falling back to simulated social media data")
    
    def get_random_user_agent(self):
        """
        Get a random user agent.
        
        Returns:
            str: Random user agent
        """
        return random.choice(self.user_agents)
    
    def clean_text(self, text):
        """
        Clean text for sentiment analysis.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        text = re.sub(r'http\S+', '', text)
        
        text = re.sub(r'<.*?>', '', text)
        
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        text = text.lower()
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            tuple: (polarity, subjectivity)
        """
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text:
            return 0, 0
        
        blob = TextBlob(cleaned_text)
        
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    
    def fetch_news_articles(self, symbol, days=7):
        """
        Fetch news articles for a symbol.
        
        Args:
            symbol (str): Symbol to fetch news for
            days (int): Number of days to look back
            
        Returns:
            list: List of news articles
        """
        logger.info(f"Fetching news articles for {symbol}")
        
        search_term = symbol.split('-')[0].lower()  # Extract BTC from BTC-USDT-VANILLA-PERPETUAL
        
        articles = []
        
        for source in self.news_sources:
            try:
                time.sleep(random.uniform(1, 3))
                
                headers = {'User-Agent': self.get_random_user_agent()}
                response = requests.get(source, headers=headers, timeout=10)
                
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch {source}: {response.status_code}")
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                article_elements = soup.find_all('article') or soup.find_all('div', class_='article')
                
                for article in article_elements[:10]:  # Limit to 10 articles per source
                    title_element = article.find('h2') or article.find('h3')
                    if not title_element:
                        continue
                    
                    title = title_element.text.strip()
                    
                    if search_term.lower() in title.lower():
                        content_element = article.find('p')
                        content = content_element.text.strip() if content_element else ''
                        
                        date_element = article.find('time') or article.find('span', class_='date')
                        date = date_element.text.strip() if date_element else datetime.now().strftime('%Y-%m-%d')
                        
                        articles.append({
                            'source': source,
                            'title': title,
                            'content': content,
                            'date': date
                        })
            
            except Exception as e:
                logger.error(f"Error fetching news from {source}: {e}")
        
        logger.info(f"Found {len(articles)} news articles for {symbol}")
        
        return articles
    
    def fetch_social_media_posts(self, symbol, days=7):
        """
        Fetch social media posts for a symbol using Twitter API.
        
        Args:
            symbol (str): Symbol to fetch posts for
            days (int): Number of days to look back
            
        Returns:
            list: List of social media posts
        """
        logger.info(f"Fetching social media posts for {symbol}")
        
        search_term = symbol.split('-')[0].lower()  # Extract BTC from BTC-USDT-VANILLA-PERPETUAL
        
        posts = []
        
        if self.twitter_client_v2 is None:
            logger.warning(f"Twitter API not initialized. Using simulated data for {symbol}")
            return self._generate_simulated_posts(search_term, days)
        
        try:
            if search_term.lower() == 'btc':
                search_queries = ["bitcoin", "btc", "#bitcoin", "#btc", "bitcoin price", "btc price"]
            elif search_term.lower() == 'eth':
                search_queries = ["ethereum", "eth", "#ethereum", "#eth", "ethereum price", "eth price"]
            elif search_term.lower() == 'sui':
                search_queries = ["sui", "#sui", "sui coin", "sui crypto", "sui price", "#suicoin"]
            else:
                search_queries = [search_term, f"#{search_term}", f"{search_term} price", f"{search_term} crypto"]
            
            start_time = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')
            
            for query in search_queries:
                try:
                    response = self.twitter_client_v2.search_recent_tweets(
                        query=query,
                        max_results=100,
                        start_time=start_time,
                        tweet_fields=['created_at', 'public_metrics', 'lang']
                    )
                    
                    if response.data:
                        for tweet in response.data:
                            if hasattr(tweet, 'lang') and tweet.lang != 'en':
                                continue
                                
                            tweet_text = tweet.text
                            created_at = tweet.created_at.strftime('%Y-%m-%d')
                            
                            engagement = 1.0
                            if hasattr(tweet, 'public_metrics'):
                                likes = tweet.public_metrics.get('like_count', 0)
                                retweets = tweet.public_metrics.get('retweet_count', 0)
                                replies = tweet.public_metrics.get('reply_count', 0)
                                
                                # Calculate engagement score (simple version)
                                engagement = 1.0 + (likes * 0.5 + retweets * 1.0 + replies * 0.3) / 100
                                engagement = min(engagement, 5.0)  # Cap at 5x
                            
                            # Analyze sentiment
                            polarity, subjectivity = self.analyze_sentiment(tweet_text)
                            
                            weighted_polarity = polarity * engagement
                            
                            posts.append({
                                'source': 'twitter',
                                'content': tweet_text,
                                'date': created_at,
                                'sentiment': (weighted_polarity, subjectivity),
                                'engagement': engagement
                            })
                    
                    time.sleep(2)  # Respect rate limits
                
                except Exception as e:
                    logger.error(f"Error fetching tweets for query '{query}': {e}")
            
            logger.info(f"Found {len(posts)} Twitter posts for {symbol}")
            
            if not posts:
                logger.warning(f"No Twitter posts found for {symbol}. Using simulated data.")
                return self._generate_simulated_posts(search_term, days)
            
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching social media posts: {e}")
            logger.warning(f"Falling back to simulated data for {symbol}")
            return self._generate_simulated_posts(search_term, days)
    
    def _generate_simulated_posts(self, search_term, days=7):
        """
        Generate simulated social media posts when Twitter API is unavailable.
        
        Args:
            search_term (str): Search term for the cryptocurrency
            days (int): Number of days to look back
            
        Returns:
            list: List of simulated social media posts
        """
        logger.info(f"Generating simulated social media posts for {search_term}")
        
        posts = []
        
        for _ in range(50):  # 50 simulated posts
            days_ago = random.randint(0, days - 1)
            post_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
            
            source = random.choice(self.social_sources)
            
            sentiment_bias = 0.2 if random.random() > 0.5 else -0.2
            
            if random.random() > 0.7:  # 30% bullish
                content = f"Bullish on {search_term.upper()}! Looking strong on the charts. #crypto #bullish"
                sentiment = (0.7 + random.random() * 0.3 + sentiment_bias, 0.6)
            elif random.random() > 0.6:  # 30% bearish
                content = f"Bearish on {search_term.upper()} in the short term. Be careful. #crypto #bearish"
                sentiment = (-0.7 - random.random() * 0.3 + sentiment_bias, 0.6)
            else:  # 40% neutral
                content = f"Watching {search_term.upper()} closely. Interesting price action. #crypto #trading"
                sentiment = (random.random() * 0.4 - 0.2 + sentiment_bias, 0.4)
            
            posts.append({
                'source': source,
                'content': content,
                'date': post_date,
                'sentiment': sentiment
            })
        
        logger.info(f"Generated {len(posts)} simulated social media posts for {search_term}")
        
        return posts
    
    def calculate_sentiment_scores(self, symbol):
        """
        Calculate sentiment scores for a symbol.
        
        Args:
            symbol (str): Symbol to calculate sentiment for
            
        Returns:
            dict: Sentiment scores
        """
        logger.info(f"Calculating sentiment scores for {symbol}")
        
        articles = self.fetch_news_articles(symbol, self.lookback_days)
        
        posts = self.fetch_social_media_posts(symbol, self.lookback_days)
        
        news_sentiments = []
        
        for article in articles:
            text = article['title'] + ' ' + article['content']
            
            polarity, subjectivity = self.analyze_sentiment(text)
            
            news_sentiments.append({
                'date': article['date'],
                'polarity': polarity,
                'subjectivity': subjectivity
            })
        
        social_sentiments = []
        
        total_engagement = 0
        weighted_social_polarity_sum = 0
        weighted_social_subjectivity_sum = 0
        
        for post in posts:
            if 'sentiment' in post:
                polarity, subjectivity = post['sentiment']
            else:
                polarity, subjectivity = self.analyze_sentiment(post['content'])
            
            engagement = post.get('engagement', 1.0)
            total_engagement += engagement
            
            weighted_social_polarity_sum += polarity * engagement
            weighted_social_subjectivity_sum += subjectivity * engagement
            
            social_sentiments.append({
                'date': post['date'],
                'polarity': polarity,
                'subjectivity': subjectivity,
                'engagement': engagement
            })
        
        if news_sentiments:
            avg_news_polarity = np.mean([s['polarity'] for s in news_sentiments])
            avg_news_subjectivity = np.mean([s['subjectivity'] for s in news_sentiments])
        else:
            avg_news_polarity = 0
            avg_news_subjectivity = 0
        
        if social_sentiments:
            if total_engagement > len(social_sentiments):
                avg_social_polarity = weighted_social_polarity_sum / total_engagement
                avg_social_subjectivity = weighted_social_subjectivity_sum / total_engagement
            else:
                avg_social_polarity = np.mean([s['polarity'] for s in social_sentiments])
                avg_social_subjectivity = np.mean([s['subjectivity'] for s in social_sentiments])
        else:
            avg_social_polarity = 0
            avg_social_subjectivity = 0
        
        if news_sentiments and social_sentiments:
            news_weight = 0.6
            social_weight = 0.4
        elif news_sentiments:
            news_weight = 1.0
            social_weight = 0.0
        elif social_sentiments:
            news_weight = 0.0
            social_weight = 1.0
        else:
            news_weight = 0.0
            social_weight = 0.0
        
        weighted_polarity = (avg_news_polarity * news_weight) + (avg_social_polarity * social_weight)
        weighted_subjectivity = (avg_news_subjectivity * news_weight) + (avg_social_subjectivity * social_weight)
        
        sentiment_score = weighted_polarity * 100
        
        sentiment_strength = abs(weighted_polarity) * 100
        
        sentiment_confidence = (1 - weighted_subjectivity) * 100
        
        # Calculate sentiment distribution
        if social_sentiments:
            sentiment_counts = Counter()
            for s in social_sentiments:
                if s['polarity'] > 0.3:
                    sentiment_counts['bullish'] += 1
                elif s['polarity'] < -0.3:
                    sentiment_counts['bearish'] += 1
                else:
                    sentiment_counts['neutral'] += 1
                    
            total = sum(sentiment_counts.values())
            sentiment_distribution = {
                'bullish_pct': (sentiment_counts['bullish'] / total * 100) if total > 0 else 0,
                'bearish_pct': (sentiment_counts['bearish'] / total * 100) if total > 0 else 0,
                'neutral_pct': (sentiment_counts['neutral'] / total * 100) if total > 0 else 0
            }
        else:
            sentiment_distribution = {
                'bullish_pct': 0,
                'bearish_pct': 0,
                'neutral_pct': 0
            }
        
        if sentiment_score > 30:
            sentiment_label = 'Very Bullish'
        elif sentiment_score > 10:
            sentiment_label = 'Bullish'
        elif sentiment_score > -10:
            sentiment_label = 'Neutral'
        elif sentiment_score > -30:
            sentiment_label = 'Bearish'
        else:
            sentiment_label = 'Very Bearish'
        
        sentiment_scores = {
            'symbol': symbol,
            'sentiment_score': sentiment_score,
            'sentiment_strength': sentiment_strength,
            'sentiment_confidence': sentiment_confidence,
            'sentiment_label': sentiment_label,
            'news_sentiment': avg_news_polarity * 100,
            'social_sentiment': avg_social_polarity * 100,
            'news_count': len(news_sentiments),
            'social_count': len(social_sentiments),
            'sentiment_distribution': sentiment_distribution,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"Sentiment scores for {symbol}: {sentiment_scores}")
        
        return sentiment_scores
    
    def calculate_sentiment_history(self, symbol, days=7):
        """
        Calculate sentiment history for a symbol.
        
        Args:
            symbol (str): Symbol to calculate sentiment for
            days (int): Number of days to look back
            
        Returns:
            pandas.DataFrame: Sentiment history
        """
        logger.info(f"Calculating sentiment history for {symbol}")
        
        articles = self.fetch_news_articles(symbol, days)
        
        posts = self.fetch_social_media_posts(symbol, days)
        
        sentiment_by_date = {}
        
        for article in articles:
            date = article['date']
            
            if date not in sentiment_by_date:
                sentiment_by_date[date] = {
                    'news_polarities': [],
                    'news_subjectivities': [],
                    'social_polarities': [],
                    'social_subjectivities': [],
                    'social_engagements': [],
                    'bullish_count': 0,
                    'neutral_count': 0,
                    'bearish_count': 0
                }
            
            text = article['title'] + ' ' + article['content']
            
            polarity, subjectivity = self.analyze_sentiment(text)
            
            sentiment_by_date[date]['news_polarities'].append(polarity)
            sentiment_by_date[date]['news_subjectivities'].append(subjectivity)
            
            # Categorize news sentiment
            if polarity > 0.3:
                sentiment_by_date[date]['bullish_count'] += 1
            elif polarity < -0.3:
                sentiment_by_date[date]['bearish_count'] += 1
            else:
                sentiment_by_date[date]['neutral_count'] += 1
        
        for post in posts:
            date = post['date']
            
            if date not in sentiment_by_date:
                sentiment_by_date[date] = {
                    'news_polarities': [],
                    'news_subjectivities': [],
                    'social_polarities': [],
                    'social_subjectivities': [],
                    'social_engagements': [],
                    'bullish_count': 0,
                    'neutral_count': 0,
                    'bearish_count': 0
                }
            
            if 'sentiment' in post:
                polarity, subjectivity = post['sentiment']
            else:
                polarity, subjectivity = self.analyze_sentiment(post['content'])
            
            engagement = post.get('engagement', 1.0)
            
            sentiment_by_date[date]['social_polarities'].append(polarity)
            sentiment_by_date[date]['social_subjectivities'].append(subjectivity)
            sentiment_by_date[date]['social_engagements'].append(engagement)
            
            # Categorize social sentiment
            if polarity > 0.3:
                sentiment_by_date[date]['bullish_count'] += 1
            elif polarity < -0.3:
                sentiment_by_date[date]['bearish_count'] += 1
            else:
                sentiment_by_date[date]['neutral_count'] += 1
        
        daily_sentiment = []
        
        for date, sentiments in sentiment_by_date.items():
            if sentiments['news_polarities']:
                avg_news_polarity = np.mean(sentiments['news_polarities'])
                avg_news_subjectivity = np.mean(sentiments['news_subjectivities'])
            else:
                avg_news_polarity = 0
                avg_news_subjectivity = 0
            
            if sentiments['social_polarities']:
                if sum(sentiments['social_engagements']) > len(sentiments['social_polarities']):
                    weighted_social_polarity_sum = sum([p * e for p, e in zip(sentiments['social_polarities'], sentiments['social_engagements'])])
                    weighted_social_subjectivity_sum = sum([s * e for s, e in zip(sentiments['social_subjectivities'], sentiments['social_engagements'])])
                    total_engagement = sum(sentiments['social_engagements'])
                    
                    avg_social_polarity = weighted_social_polarity_sum / total_engagement
                    avg_social_subjectivity = weighted_social_subjectivity_sum / total_engagement
                else:
                    avg_social_polarity = np.mean(sentiments['social_polarities'])
                    avg_social_subjectivity = np.mean(sentiments['social_subjectivities'])
            else:
                avg_social_polarity = 0
                avg_social_subjectivity = 0
            
            if sentiments['news_polarities'] and sentiments['social_polarities']:
                news_weight = 0.6
                social_weight = 0.4
            elif sentiments['news_polarities']:
                news_weight = 1.0
                social_weight = 0.0
            elif sentiments['social_polarities']:
                news_weight = 0.0
                social_weight = 1.0
            else:
                news_weight = 0.0
                social_weight = 0.0
            
            weighted_polarity = (avg_news_polarity * news_weight) + (avg_social_polarity * social_weight)
            weighted_subjectivity = (avg_news_subjectivity * news_weight) + (avg_social_subjectivity * social_weight)
            
            sentiment_score = weighted_polarity * 100
            
            # Calculate sentiment distribution percentages
            total_count = sentiments['bullish_count'] + sentiments['neutral_count'] + sentiments['bearish_count']
            
            if total_count > 0:
                bullish_pct = sentiments['bullish_count'] / total_count * 100
                neutral_pct = sentiments['neutral_count'] / total_count * 100
                bearish_pct = sentiments['bearish_count'] / total_count * 100
            else:
                bullish_pct = 0
                neutral_pct = 0
                bearish_pct = 0
            
            daily_sentiment.append({
                'date': date,
                'sentiment_score': sentiment_score,
                'news_sentiment': avg_news_polarity * 100,
                'social_sentiment': avg_social_polarity * 100,
                'news_count': len(sentiments['news_polarities']),
                'social_count': len(sentiments['social_polarities']),
                'bullish_pct': bullish_pct,
                'neutral_pct': neutral_pct,
                'bearish_pct': bearish_pct
            })
        
        df = pd.DataFrame(daily_sentiment)
        
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        logger.info(f"Sentiment history for {symbol}: {len(df)} days")
        
        return df
    
    def plot_sentiment_history(self, symbol, sentiment_history):
        """
        Plot sentiment history for a symbol.
        
        Args:
            symbol (str): Symbol to plot sentiment for
            sentiment_history (pandas.DataFrame): Sentiment history
        """
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        plt.figure(figsize=(12, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(sentiment_history['date'], sentiment_history['sentiment_score'], label='Overall Sentiment', linewidth=2)
        plt.plot(sentiment_history['date'], sentiment_history['news_sentiment'], label='News Sentiment', linestyle='--')
        plt.plot(sentiment_history['date'], sentiment_history['social_sentiment'], label='Social Sentiment', linestyle=':')
        plt.title(f'Sentiment History for {symbol}')
        plt.ylabel('Sentiment Score (-100 to 100)')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        plt.grid(True)
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.bar(sentiment_history['date'], sentiment_history['news_count'], label='News Count')
        plt.bar(sentiment_history['date'], sentiment_history['social_count'], bottom=sentiment_history['news_count'], label='Social Count')
        plt.title('Content Volume')
        plt.ylabel('Count')
        plt.grid(True)
        plt.legend()
        
        if 'bullish_pct' in sentiment_history.columns:
            plt.subplot(3, 1, 3)
            
            # Create stacked area chart for sentiment distribution
            dates = sentiment_history['date']
            bullish = sentiment_history['bullish_pct']
            neutral = sentiment_history['neutral_pct']
            bearish = sentiment_history['bearish_pct']
            
            plt.stackplot(dates, 
                         [bullish, neutral, bearish],
                         labels=['Bullish', 'Neutral', 'Bearish'],
                         colors=['green', 'gray', 'red'],
                         alpha=0.7)
            
            plt.title('Sentiment Distribution')
            plt.ylabel('Percentage')
            plt.ylim(0, 100)
            plt.grid(True)
            plt.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/sentiment_history_{symbol}_{timestamp}.png')
        
        sentiment_history.to_csv(f'{results_dir}/sentiment_history_{symbol}_{timestamp}.csv', index=False)
        
        logger.info(f"Sentiment history plot saved to {results_dir}")
    
    def run_sentiment_analysis(self):
        """
        Run sentiment analysis for all symbols.
        
        Returns:
            dict: Sentiment scores for all symbols
        """
        logger.info("Running sentiment analysis")
        
        sentiment_scores = {}
        
        for symbol in self.symbols:
            scores = self.calculate_sentiment_scores(symbol)
            
            history = self.calculate_sentiment_history(symbol, self.lookback_days)
            
            if not history.empty:
                self.plot_sentiment_history(symbol, history)
            
            sentiment_scores[symbol] = scores
        
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with open(f'{results_dir}/sentiment_scores_{timestamp}.json', 'w') as f:
            json.dump(sentiment_scores, f, indent=4)
        
        logger.info(f"Sentiment analysis complete: {len(sentiment_scores)} symbols")
        
        return sentiment_scores

def integrate_sentiment_with_strategy(df, symbol, sentiment_score=None):
    """
    Integrate sentiment analysis with trading strategy.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data and indicators
        symbol (str): Symbol to integrate sentiment for
        sentiment_score (float): Sentiment score (-100 to 100)
        
    Returns:
        pandas.DataFrame: DataFrame with sentiment-adjusted signals
    """
    logger.info(f"Integrating sentiment with strategy for {symbol}")
    
    if sentiment_score is None:
        analyzer = SentimentAnalyzer([symbol])
        scores = analyzer.calculate_sentiment_scores(symbol)
        sentiment_score = scores['sentiment_score']
    
    normalized_sentiment = sentiment_score / 100
    
    df['sentiment_score'] = normalized_sentiment
    
    if 'combined_buy' in df.columns:
        if normalized_sentiment > 0.3:
            df['sentiment_adjusted_buy'] = df['combined_buy'] | (df['rsi'] < 40) & (df['macd_hist'] > 0)
        elif normalized_sentiment < -0.3:
            df['sentiment_adjusted_buy'] = df['combined_buy'] & (df['rsi'] < 30) & (df['macd_hist'] > 0)
        else:
            df['sentiment_adjusted_buy'] = df['combined_buy']
    
    if 'combined_sell' in df.columns:
        if normalized_sentiment < -0.3:
            df['sentiment_adjusted_sell'] = df['combined_sell'] | (df['rsi'] > 60) & (df['macd_hist'] < 0)
        elif normalized_sentiment > 0.3:
            df['sentiment_adjusted_sell'] = df['combined_sell'] & (df['rsi'] > 70) & (df['macd_hist'] < 0)
        else:
            df['sentiment_adjusted_sell'] = df['combined_sell']
    
    if 'hedge_fund_buy' in df.columns:
        if normalized_sentiment > 0.3:
            df['sentiment_adjusted_hf_buy'] = df['hedge_fund_buy'] | (df['rsi'] < 40) & (df['macd_hist'] > 0)
        elif normalized_sentiment < -0.3:
            df['sentiment_adjusted_hf_buy'] = df['hedge_fund_buy'] & (df['rsi'] < 30) & (df['macd_hist'] > 0)
        else:
            df['sentiment_adjusted_hf_buy'] = df['hedge_fund_buy']
    
    if 'hedge_fund_sell' in df.columns:
        if normalized_sentiment < -0.3:
            df['sentiment_adjusted_hf_sell'] = df['hedge_fund_sell'] | (df['rsi'] > 60) & (df['macd_hist'] < 0)
        elif normalized_sentiment > 0.3:
            df['sentiment_adjusted_hf_sell'] = df['hedge_fund_sell'] & (df['rsi'] > 70) & (df['macd_hist'] < 0)
        else:
            df['sentiment_adjusted_hf_sell'] = df['hedge_fund_sell']
    
    logger.info(f"Sentiment integration complete for {symbol}")
    
    return df

def run_sentiment_backtest(symbol='BTC-USDT-VANILLA-PERPETUAL', interval='1h'):
    """
    Run backtest with sentiment-adjusted signals.
    
    Args:
        symbol (str): Symbol to run backtest for
        interval (str): Timeframe interval
        
    Returns:
        tuple: (metrics_without_sentiment, metrics_with_sentiment)
    """
    logger.info(f"Running sentiment backtest for {symbol} {interval}")
    
    from coindesk_client import get_data
    from indicators import add_indicators
    from squeeze_momentum import add_squeeze_momentum
    from ultimate_macd import add_ultimate_macd
    from generic_support_resistance import detect_support_resistance_levels
    from combined_strategy import generate_combined_signals
    from hedge_fund_strategy import generate_hedge_fund_signals
    
    df = get_data(symbol, 'hours', interval)
    df = add_indicators(df)
    df = add_squeeze_momentum(df)
    df = add_ultimate_macd(df)
    df = detect_support_resistance_levels(df, symbol, interval)
    df = generate_combined_signals(df, symbol, interval)
    df = generate_hedge_fund_signals(df, symbol, interval)
    
    from hedge_fund_strategy import run_hedge_fund_backtest
    metrics_without_sentiment = run_hedge_fund_backtest(df, symbol, interval)
    
    analyzer = SentimentAnalyzer([symbol])
    sentiment_history = analyzer.calculate_sentiment_history(symbol)
    
    df_with_sentiment = integrate_sentiment_with_strategy(df.copy(), symbol)
    
    def run_sentiment_adjusted_backtest(df, initial_capital=10000.0):
        """
        Run backtest with sentiment-adjusted signals.
        
        Args:
            df (pandas.DataFrame): DataFrame with sentiment-adjusted signals
            initial_capital (float): Initial capital
            
        Returns:
            dict: Backtest metrics
        """
        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = [capital]
        
        for i in range(1, len(df)):
            if pd.isna(df['close'].iloc[i]):
                equity_curve.append(equity_curve[-1])
                continue
            
            if position > 0 and df['sentiment_adjusted_hf_sell'].iloc[i]:
                exit_price = df['close'].iloc[i]
                profit_loss = position * (exit_price - entry_price)
                capital += profit_loss
                trades.append({
                    'entry_date': df.index[i - position],
                    'exit_date': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': (exit_price / entry_price - 1) * 100,
                    'type': 'long'
                })
                position = 0
            
            if position == 0 and df['sentiment_adjusted_hf_buy'].iloc[i]:
                entry_price = df['close'].iloc[i]
                position_capital = capital * 0.95  # Use 95% of capital
                position = position_capital / entry_price
                position = i  # Store position index
            
            if position > 0:
                current_price = df['close'].iloc[i]
                current_value = capital + position * (current_price - entry_price)
            else:
                current_value = capital
            
            equity_curve.append(current_value)
        
        if position > 0:
            exit_price = df['close'].iloc[-1]
            profit_loss = position * (exit_price - entry_price)
            capital += profit_loss
            trades.append({
                'entry_date': df.index[len(df) - position],
                'exit_date': df.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit_loss': profit_loss,
                'profit_loss_pct': (exit_price / entry_price - 1) * 100,
                'type': 'long'
            })
        
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'total_return': 0,
                'annualized_return': 0
            }
        
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['profit_loss'] > 0]
        losing_trades = [t for t in trades if t['profit_loss'] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        avg_profit = np.mean([t['profit_loss_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit_loss_pct'] for t in losing_trades]) if losing_trades else 0
        
        total_profit = sum([t['profit_loss'] for t in winning_trades])
        total_loss = abs(sum([t['profit_loss'] for t in losing_trades]))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        peak = initial_capital
        drawdowns = []
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            drawdowns.append(drawdown)
        
        max_drawdown = max(drawdowns)
        
        total_return = (capital / initial_capital - 1) * 100
        
        days = (trades[-1]['exit_date'] - trades[0]['entry_date']).days
        annualized_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100 if days > 0 else 0
        
        equity_returns = [equity_curve[i] / equity_curve[i-1] - 1 for i in range(1, len(equity_curve))]
        sharpe_ratio = np.mean(equity_returns) / np.std(equity_returns) * np.sqrt(252) if np.std(equity_returns) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'annualized_return': annualized_return
        }
    
    metrics_with_sentiment = run_sentiment_adjusted_backtest(df_with_sentiment)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.bar(['Without Sentiment', 'With Sentiment'], [metrics_without_sentiment['total_return'], metrics_with_sentiment['total_return']])
    plt.title('Total Return (%)')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.bar(['Without Sentiment', 'With Sentiment'], [metrics_without_sentiment['win_rate'], metrics_with_sentiment['win_rate']])
    plt.title('Win Rate (%)')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.bar(['Without Sentiment', 'With Sentiment'], [metrics_without_sentiment['max_drawdown'], metrics_with_sentiment['max_drawdown']])
    plt.title('Max Drawdown (%)')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.bar(['Without Sentiment', 'With Sentiment'], [metrics_without_sentiment['sharpe_ratio'], metrics_with_sentiment['sharpe_ratio']])
    plt.title('Sharpe Ratio')
    plt.grid(True)
    
    plt.tight_layout()
    
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    plt.savefig(f'{results_dir}/sentiment_backtest_comparison_{symbol}_{interval}_{timestamp}.png')
    
    comparison = {
        'without_sentiment': metrics_without_sentiment,
        'with_sentiment': metrics_with_sentiment,
        'improvement': {
            'total_return': metrics_with_sentiment['total_return'] - metrics_without_sentiment['total_return'],
            'win_rate': metrics_with_sentiment['win_rate'] - metrics_without_sentiment['win_rate'],
            'max_drawdown': metrics_with_sentiment['max_drawdown'] - metrics_without_sentiment['max_drawdown'],
            'sharpe_ratio': metrics_with_sentiment['sharpe_ratio'] - metrics_without_sentiment['sharpe_ratio']
        }
    }
    
    with open(f'{results_dir}/sentiment_backtest_comparison_{symbol}_{interval}_{timestamp}.json', 'w') as f:
        json.dump(comparison, f, indent=4)
    
    logger.info(f"Sentiment backtest complete: {comparison['improvement']}")
    
    return metrics_without_sentiment, metrics_with_sentiment

if __name__ == "__main__":
    analyzer = SentimentAnalyzer(['BTC-USDT-VANILLA-PERPETUAL', 'SUI-USDT-VANILLA-PERPETUAL'])
    sentiment_scores = analyzer.run_sentiment_analysis()
    
    btc_metrics = run_sentiment_backtest('BTC-USDT-VANILLA-PERPETUAL', '1h')
    sui_metrics = run_sentiment_backtest('SUI-USDT-VANILLA-PERPETUAL', '1h')
    
    logger.info("Sentiment analysis and backtest complete")
