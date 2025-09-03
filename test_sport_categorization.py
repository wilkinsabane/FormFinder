#!/usr/bin/env python3
"""Test script to investigate sport categorization in RSS feeds."""

import sys
sys.path.append('.')

from formfinder.rss_news_provider import create_default_rss_provider
from formfinder.rss_content_parser import ParsedArticle

def main():
    # Create provider
    provider = create_default_rss_provider()
    
    # Check the sport keywords in the parser config
    print('Sport keywords in parser config:')
    for sport, keywords in provider.content_parser.config.sport_keywords.items():
        print(f'  {sport}: {keywords}')
    
    # Check a few RSS feeds and their sport settings
    print('\nRSS feeds and their sport settings:')
    for i, feed in enumerate(provider.feed_manager.config.feeds[:5]):
        print(f'  {feed.name}: sport={feed.sport}, league={feed.league}')
    
    # Test parsing a sample article
    sample_article = ParsedArticle(
        title='Manchester United beats Arsenal in Premier League match',
        content='Great football match between two top teams in the Premier League.',
        url='http://example.com',
        published_date=None,
        sport='',  # Empty initially
        language='en'
    )
    
    print('\nBefore categorization:')
    print(f'  Article sport: "{sample_article.sport}"')
    
    provider.content_parser._categorize_article(sample_article)
    
    print('After categorization:')
    print(f'  Article sport: "{sample_article.sport}"')
    
    # Test with different content that might not match
    sample_article2 = ParsedArticle(
        title='Bundesliga match report',
        content='Bayern Munich played against Borussia Dortmund in a thrilling Bundesliga encounter.',
        url='http://example.com/2',
        published_date=None,
        sport='',
        language='en'
    )
    
    print('\nTesting Bundesliga article:')
    print(f'  Before: "{sample_article2.sport}"')
    provider.content_parser._categorize_article(sample_article2)
    print(f'  After: "{sample_article2.sport}"')
    
    # Test with content that doesn't match any sport
    sample_article3 = ParsedArticle(
        title='Weather report for today',
        content='It will be sunny with temperatures reaching 25 degrees.',
        url='http://example.com/3',
        published_date=None,
        sport='',
        language='en'
    )
    
    print('\nTesting non-sport article:')
    print(f'  Before: "{sample_article3.sport}"')
    provider.content_parser._categorize_article(sample_article3)
    print(f'  After: "{sample_article3.sport}"')

if __name__ == '__main__':
    main()