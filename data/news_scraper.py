"""
News Scraper - Collects financial news for sentiment analysis.
Fetches from RSS feeds, Google News, and financial portals.
"""
import hashlib
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import feedparser
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class NewsArticle:
    """Represents a single news article."""

    def __init__(self, title: str, content: str, source: str, published: datetime,
                 url: str = "", symbols: Optional[List[str]] = None):
        self.title = title
        self.content = content
        self.source = source
        self.published = published
        self.url = url
        self.symbols = symbols or []
        self.sentiment_score: Optional[float] = None
        self.sentiment_label: Optional[str] = None
        self.id = hashlib.sha256(f"{title}{url}".encode()).hexdigest()[:16]


class NewsScraper:
    """Scrapes financial news from multiple sources."""

    # Map of common NSE stock names to symbols for entity extraction
    COMPANY_SYMBOL_MAP = {
        "reliance": "RELIANCE", "tcs": "TCS", "infosys": "INFY",
        "hdfc bank": "HDFCBANK", "icici bank": "ICICIBANK", "sbi": "SBIN",
        "state bank": "SBIN", "bharti airtel": "BHARTIARTL", "airtel": "BHARTIARTL",
        "itc": "ITC", "kotak": "KOTAKBANK", "l&t": "LT", "larsen": "LT",
        "axis bank": "AXISBANK", "asian paints": "ASIANPAINT", "maruti": "MARUTI",
        "titan": "TITAN", "sun pharma": "SUNPHARMA", "bajaj finance": "BAJFINANCE",
        "wipro": "WIPRO", "hcl tech": "HCLTECH", "ntpc": "NTPC",
        "ongc": "ONGC", "power grid": "POWERGRID", "tata motors": "TATAMOTORS",
        "jsw steel": "JSWSTEEL", "tata steel": "TATASTEEL", "adani": "ADANIENT",
        "ultratech": "ULTRACEMCO", "nestle": "NESTLEIND", "tech mahindra": "TECHM",
        "mahindra": "M&M", "indusind": "INDUSINDBK", "coal india": "COALINDIA",
        "cipla": "CIPLA", "dr reddy": "DRREDDY", "britannia": "BRITANNIA",
        "apollo hospital": "APOLLOHOSP", "eicher": "EICHERMOT",
        "hero motocorp": "HEROMOTOCO", "tata consumer": "TATACONSUM",
        "hindalco": "HINDALCO", "bajaj auto": "BAJAJ-AUTO", "bpcl": "BPCL",
        "nifty": "_NIFTY", "sensex": "_SENSEX", "bank nifty": "_BANKNIFTY",
    }

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    def __init__(self, config):
        self.config = config
        self._seen_ids = set()
        self._articles_today: List[NewsArticle] = []

    def fetch_rss_feeds(self) -> List[NewsArticle]:
        """Fetch articles from configured RSS feeds."""
        articles = []
        for feed_url in self.config.news.sources:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:self.config.news.max_articles_per_fetch]:
                    published = datetime.now()
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        published = datetime(*entry.published_parsed[:6])

                    content = entry.get("summary", entry.get("description", ""))
                    # Strip HTML tags
                    content = BeautifulSoup(content, "html.parser").get_text()

                    article = NewsArticle(
                        title=entry.get("title", ""),
                        content=content,
                        source=feed_url,
                        published=published,
                        url=entry.get("link", ""),
                    )
                    article.symbols = self._extract_symbols(
                        f"{article.title} {article.content}"
                    )

                    if article.id not in self._seen_ids:
                        self._seen_ids.add(article.id)
                        articles.append(article)

            except Exception as e:
                logger.error(f"RSS feed error ({feed_url}): {e}")

        logger.info(f"Fetched {len(articles)} new articles from RSS feeds")
        return articles

    def fetch_google_news(self, query: Optional[str] = None) -> List[NewsArticle]:
        """Fetch news from Google News RSS."""
        if query is None:
            query = self.config.news.google_news_query

        articles = []
        try:
            url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(url)

            for entry in feed.entries[:self.config.news.max_articles_per_fetch]:
                published = datetime.now()
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6])

                article = NewsArticle(
                    title=entry.get("title", ""),
                    content=entry.get("summary", ""),
                    source="google_news",
                    published=published,
                    url=entry.get("link", ""),
                )
                article.symbols = self._extract_symbols(article.title)

                if article.id not in self._seen_ids:
                    self._seen_ids.add(article.id)
                    articles.append(article)

        except Exception as e:
            logger.error(f"Google News error: {e}")

        logger.info(f"Fetched {len(articles)} articles from Google News")
        return articles

    def fetch_stock_specific_news(self, symbol: str) -> List[NewsArticle]:
        """Fetch news for a specific stock symbol."""
        query = f"{symbol} NSE stock India"
        return self.fetch_google_news(query=query)

    def fetch_all_news(self) -> List[NewsArticle]:
        """Fetch news from all sources."""
        all_articles = []
        all_articles.extend(self.fetch_rss_feeds())
        all_articles.extend(self.fetch_google_news())

        # Also search for major market keywords
        for query in ["NSE India market", "Indian stock market rally crash"]:
            all_articles.extend(self.fetch_google_news(query=query))
            time.sleep(1)  # Be polite

        self._articles_today.extend(all_articles)
        return all_articles

    def get_today_articles(self) -> List[NewsArticle]:
        """Get all articles collected today."""
        today = datetime.now().date()
        return [a for a in self._articles_today if a.published.date() == today]

    def get_symbol_sentiment_summary(self) -> Dict[str, Dict]:
        """Get aggregated sentiment per symbol from today's articles."""
        summary = {}
        for article in self.get_today_articles():
            if article.sentiment_score is None:
                continue
            for symbol in article.symbols:
                if symbol not in summary:
                    summary[symbol] = {
                        "scores": [], "articles": 0,
                        "positive": 0, "negative": 0, "neutral": 0,
                    }
                summary[symbol]["scores"].append(article.sentiment_score)
                summary[symbol]["articles"] += 1
                if article.sentiment_label == "positive":
                    summary[symbol]["positive"] += 1
                elif article.sentiment_label == "negative":
                    summary[symbol]["negative"] += 1
                else:
                    summary[symbol]["neutral"] += 1

        # Calculate averages
        for symbol in summary:
            scores = summary[symbol]["scores"]
            summary[symbol]["avg_sentiment"] = sum(scores) / len(scores) if scores else 0
            summary[symbol]["sentiment_std"] = (
                (sum((s - summary[symbol]["avg_sentiment"]) ** 2 for s in scores) / len(scores)) ** 0.5
                if len(scores) > 1 else 0
            )

        return summary

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols mentioned in text."""
        text_lower = text.lower()
        found = set()
        for name, symbol in self.COMPANY_SYMBOL_MAP.items():
            if name in text_lower:
                found.add(symbol)
        return list(found)

    def clear_old_articles(self, days: int = 2):
        """Remove articles older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)
        before = len(self._articles_today)
        self._articles_today = [a for a in self._articles_today if a.published > cutoff]
        removed = before - len(self._articles_today)
        if removed > 0:
            logger.info(f"Cleared {removed} old articles")
