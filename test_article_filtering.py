#!/usr/bin/env python3
"""Test script to investigate article filtering in RSS feeds."""

import sys
sys.path.append('.')

from formfinder.rss_news_provider import create_default_rss_provider
from formfinder.rss_content_parser import ParsedArticle
from datetime import datetime, timedelta

def main():
    # Create provider
    provider = create_default_rss_provider()
    
    # Create test articles with different sports
    articles = [
        ParsedArticle(
            title='Manchester United beats Arsenal in Premier League match',
            content='Great football match between two top teams in the Premier League.',
            url='http://example.com/1',
            published_date=datetime.now(),
            sport='football',  # Properly categorized
            language='en'
        ),
        ParsedArticle(
            title='Bayern Munich wins Bundesliga title',
            content='Bayern Munich secured another Bundesliga championship with a victory.',
            url='http://example.com/2',
            published_date=datetime.now(),
            sport='',  # Not categorized due to missing 'bundesliga' keyword
            language='en'
        ),
        ParsedArticle(
            title='Real Madrid vs Barcelona El Clasico',
            content='The biggest match in Spanish football between Real Madrid and Barcelona.',
            url='http://example.com/3',
            published_date=datetime.now(),
            sport='football',  # Should be categorized due to 'football' keyword
            language='en'
        ),
        ParsedArticle(
            title='Serie A match report',
            content='Juventus played against AC Milan in a thrilling Serie A encounter.',
            url='http://example.com/4',
            published_date=datetime.now(),
            sport='',  # Not categorized due to missing 'serie a' keyword
            language='en'
        )
    ]
    
    print('Articles before categorization:')
    for i, article in enumerate(articles):
        print(f'  {i+1}. "{article.title}" - sport: "{article.sport}"')
    
    # Categorize articles
    for article in articles:
        provider.content_parser._categorize_article(article)
    
    print('\nArticles after categorization:')
    for i, article in enumerate(articles):
        print(f'  {i+1}. "{article.title}" - sport: "{article.sport}"')
    
    # Test filtering by sport
    print('\nTesting filter_articles_by_sport with "football":')
    football_articles = provider.content_parser.filter_articles_by_sport(articles, 'football')
    print(f'  Original articles: {len(articles)}')
    print(f'  Football articles: {len(football_articles)}')
    for article in football_articles:
        print(f'    - "{article.title}"')
    
    print('\nTesting filter_articles_by_sport with "basketball":')
    basketball_articles = provider.content_parser.filter_articles_by_sport(articles, 'basketball')
    print(f'  Basketball articles: {len(basketball_articles)}')
    
    print('\nTesting filter_articles_by_sport with empty string:')
    empty_sport_articles = provider.content_parser.filter_articles_by_sport(articles, '')
    print(f'  Empty sport articles: {len(empty_sport_articles)}')
    for article in empty_sport_articles:
        print(f'    - "{article.title}" (sport: "{article.sport}")')

if __name__ == '__main__':
    main()