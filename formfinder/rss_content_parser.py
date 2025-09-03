#!/usr/bin/env python3
"""
RSS Content Parser for Sports News Sentiment Analysis

This module provides content parsing and preprocessing capabilities including:
- RSS feed parsing and content extraction
- Text cleaning and normalization
- Content categorization and filtering
- Duplicate detection and removal
"""

import logging
import re
import hashlib
import html
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
import feedparser
from bs4 import BeautifulSoup
import requests
from dateutil import parser as date_parser

logger = logging.getLogger(__name__)

@dataclass
class ParsedArticle:
    """Parsed article from RSS feed."""
    title: str
    content: str
    url: str
    published_date: Optional[datetime] = None
    author: Optional[str] = None
    source: str = ""
    sport: str = ""
    league: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    content_hash: str = ""
    word_count: int = 0
    language: str = "en"
    quality_score: float = 0.0
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = self._generate_content_hash()
        if not self.word_count:
            self.word_count = len(self.content.split())
    
    def _generate_content_hash(self) -> str:
        """Generate unique hash for content deduplication."""
        content_for_hash = f"{self.title}|{self.content[:500]}"
        return hashlib.md5(content_for_hash.encode('utf-8')).hexdigest()

@dataclass
class ParsingConfig:
    """Configuration for RSS content parsing."""
    min_content_length: int = 50
    max_content_length: int = 10000
    extract_full_content: bool = False
    clean_html: bool = True
    normalize_text: bool = True
    filter_duplicates: bool = True
    max_articles_per_feed: int = 50
    content_timeout: int = 10
    user_agent: str = "FormFinder RSS Parser 1.0"
    
    # Content filtering patterns
    exclude_patterns: List[str] = field(default_factory=lambda: [
        r'advertisement',
        r'sponsored content',
        r'click here',
        r'subscribe now',
        r'newsletter'
    ])
    
    # Sport-specific keywords for categorization
    sport_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        'football': [
            # General terms
            'football', 'soccer', 'futbol', 'fußball', 'calcio', 'futebol',
            
            # International & Continental Competitions
            'fifa world cup', 'fifa women\'s world cup', 'world cup live', 'watch world cup',
            'world cup final tickets', 'fifa tickets', 'fifa intercontinental cup', 'fifa club world cup',
            'uefa champions league', 'ucl', 'uefa europa league', 'uefa conference league',
            'qualifying rounds', 'play-off round', 'league phase', 'knockout phase',
            'double-legged knockout', 'single-leg final', 'uefa super cup', 'uefa european championship',
            'conmebol libertadores', 'copa libertadores de américa', 'el sueño libertador',
            'la copa se mira y no se toca', 'copa sudamericana', 'concacaf champions cup',
            'leagues cup', 'concacaf world cup qualification', 'conmebol world cup qualification',
            
            # European Domestic Leagues
            # Premier League (England)
            'premier league', 'promotion', 'relegation', 'fa cup', 'efl trophy',
            'community shield', 'merseyside derby', 'northwest derby', 'mohamed salah',
            'alexander isak', 'erling braut haaland', 'shot assist',
            
            # La Liga (Spain)
            'la liga', 'primera división', 'laliga ea sports', 'liga nacional de fútbol profesional',
            'lfp', 'copa del rey', 'supercopa de españa', 'el clásico', 'trofeo pichichi',
            'kylian mbappe', 'robert lewandowski', 'lamine yamal',
            
            # Bundesliga (Germany)
            'bundesliga', '1. bundesliga', '2. bundesliga', '3. liga',
            'deutsche fußball liga', 'dfl', 'deutscher fußball-bund', 'dfb',
            'dfb-pokal', 'dfl-supercup', 'harry kane', 'michael olise',
            'jean-mattéo bahoya', 'bayern munich', 'borussia dortmund',
            
            # Serie A (Italy)
            'serie a', 'serie a enilive', 'lega serie a', 'sette sorelle',
            'juventus', 'inter milan', 'ac milan', 'coppa italia', 'supercoppa italiana',
            'derby d\'italia', 'derby della capitale', 'derby della madonnina',
            'mateo retegui', 'romelu lukaku', 'christian pulisic',
            
            # Ligue 1 (France)
            'ligue 1', 'ligue 1 mcdonald\'s', 'ligue de football professionnel',
            'coupe de france', 'trophée des champions', 'derby rhônalpin',
            'le classique', 'parisian derby', 'jonathan clauss', 'kamory doumbia',
            'mason greenwood',
            
            # Primeira Liga (Portugal)
            'primeira liga', 'liga portugal', 'liga portugal betclic',
            'taça de portugal', 'taça da liga', 'supertaça', 'big three',
            'derby de lisboa', 'viktor gyökeres', 'pedro gonçalves',
            
            # Eredivisie (Netherlands)
            'eredivisie', 'vriendenloterij eredivisie', 'ajax', 'psv', 'feyenoord',
            
            # Süper Lig (Turkey)
            'süper lig', 'trendyol süper lig', 'galatasaray', 'fenerbahçe',
            'beşiktaş', 'the intercontinental derby', 'hakan şükür',
            
            # Other European Leagues
            'danish superliga', 'superligaen', 'ekstraklasa', 'pko bank polski ekstraklasa',
            'nb i', 'fizz liga', 'super liga', 'partizan', 'red star',
            'scottish premiership', 'hnl', 'supersport hrvatska nogometna liga',
            
            # Leagues in the Americas
            # Liga Profesional (Argentina)
            'liga profesional de fútbol', 'torneo betano', 'apertura', 'clausura',
            'copa argentina', 'river plate', 'boca juniors', 'big five',
            
            # Brazilian Série A
            'campeonato brasileiro série a', 'brasileirão', 'brasileirão betano',
            'copa do brasil', 'palmeiras', 'santos', 'corinthians', 'flamengo',
            'roberto dinamite',
            
            # Liga MX (Mexico)
            'liga mx', 'liga bbva mx', 'liguilla', 'campeón de campeones',
            
            # MLS (USA)
            'major league soccer', 'mls', 'single entity', 'supporters\' shield',
            'mls cup', 'mls playoffs', 'designated player rule', 'lionel messi',
            'david beckham',
            
            # Other American Leagues
            'primera división chile', 'superclásico', 'colo-colo', 'universidad de chile',
            'primera división uruguay', 'tabla anual', 'descenso', 'primera a colombia',
            'liga betplay dimayor', 'liga pro ecuador', 'liga pro ecuabet',
            'primera division peru', 'liga 1 te apuesto', 'liga nacional honduras',
            
            # Leagues in Africa & Asia
            # Saudi League
            'saudi league', 'roshn saudi league', 'cristiano ronaldo', 'al-nassr fc', 'al-hilal',
            
            # A-League (Australia)
            'a-league men', 'melbourne derby', 'the original rivalry', 'big blue',
            
            # J1 League (Japan)
            'j1 league', 'j. league', 'xg',
            
            # K League 1 (South Korea)
            'k league 1',
            
            # Botola Pro (Morocco)
            'botola pro', 'botola pro inwi', 'royal moroccan football federation',
            
            # Egyptian Premier League
            'egyptian premier league', 'cairo derby',
            
            # Premier League (South Africa)
            'premier soccer league', 'psl', 'betway premiership', 'relegation playoff',
            
            # Other African & Asian Leagues
            'indian super league', 'isl', 'thai league 1', 'byd sealion 6 league i',
            'qatar stars league', 'qsl', 'ooredoo stars league', 'erovnuli liga',
            
            # General football terms
            'match', 'fixture', 'goal', 'penalty', 'offside', 'referee', 'stadium',
            'transfer', 'manager', 'coach', 'player', 'team', 'club', 'derby'
        ],
        'basketball': ['basketball', 'nba', 'euroleague', 'euroleague basketball', 'ncaa basketball', 'fiba', 'wnba', 'nbl', 'cba'],
        'tennis': ['tennis', 'wimbledon', 'us open', 'french open', 'australian open'],
        'cricket': ['cricket', 'test match', 'odi', 't20', 'ipl', 'county cricket'],
        'rugby': ['rugby', 'six nations', 'world cup rugby', 'premiership rugby'],
        'golf': ['golf', 'pga', 'masters', 'open championship', 'ryder cup']
    })

class RSSContentParser:
    """Parses RSS feeds and extracts content for sentiment analysis."""
    
    def __init__(self, config: ParsingConfig):
        """
        Initialize RSS content parser.
        
        Args:
            config: Parsing configuration settings
        """
        self.config = config
        self.seen_hashes: Set[str] = set()
        
        # Initialize session for full content extraction
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        })
        
        logger.info("Initialized RSS Content Parser")
    
    def parse_feed(self, feed_url: str, feed_content: Optional[str] = None) -> List[ParsedArticle]:
        """
        Parse RSS feed and extract articles.
        
        Args:
            feed_url: URL of the RSS feed
            feed_content: Pre-fetched feed content (optional)
            
        Returns:
            List of parsed articles
        """
        try:
            if feed_content:
                parsed_feed = feedparser.parse(feed_content)
            else:
                parsed_feed = feedparser.parse(feed_url)
            
            if parsed_feed.bozo:
                logger.warning(f"Feed parsing warning for {feed_url}: {parsed_feed.bozo_exception}")
            
            articles = []
            source_name = self._extract_source_name(parsed_feed, feed_url)
            
            for entry in parsed_feed.entries[:self.config.max_articles_per_feed]:
                try:
                    article = self._parse_entry(entry, source_name, feed_url)
                    if article and self._is_valid_article(article):
                        articles.append(article)
                except Exception as e:
                    logger.warning(f"Error parsing entry from {feed_url}: {e}")
                    continue
            
            logger.info(f"Parsed {len(articles)} articles from {feed_url}")
            return articles
            
        except Exception as e:
            logger.error(f"Error parsing feed {feed_url}: {e}")
            return []
    
    def _extract_source_name(self, parsed_feed, feed_url: str) -> str:
        """Extract source name from feed metadata."""
        if hasattr(parsed_feed.feed, 'title') and parsed_feed.feed.title:
            return parsed_feed.feed.title.strip()
        
        # Fallback to domain name
        try:
            domain = urlparse(feed_url).netloc
            return domain.replace('www.', '').replace('.com', '').replace('.co.uk', '')
        except:
            return "Unknown Source"
    
    def _parse_entry(self, entry, source_name: str, feed_url: str) -> Optional[ParsedArticle]:
        """
        Parse individual RSS entry into ParsedArticle.
        
        Args:
            entry: RSS entry from feedparser
            source_name: Name of the news source
            feed_url: URL of the RSS feed
            
        Returns:
            Parsed article or None if parsing fails
        """
        try:
            # Extract basic information
            title = self._clean_text(getattr(entry, 'title', ''))
            if not title:
                return None
            
            # Extract content
            content = self._extract_content(entry)
            if not content:
                return None
            
            # Extract URL
            url = getattr(entry, 'link', '')
            if not url:
                return None
            
            # Extract publication date
            published_date = self._extract_date(entry)
            
            # Extract author
            author = self._extract_author(entry)
            
            # Create article
            article = ParsedArticle(
                title=title,
                content=content,
                url=url,
                published_date=published_date,
                author=author,
                source=source_name
            )
            
            # Categorize content
            self._categorize_article(article)
            
            # Calculate quality score
            article.quality_score = self._calculate_quality_score(article)
            
            return article
            
        except Exception as e:
            logger.warning(f"Error parsing entry: {e}")
            return None
    
    def _extract_content(self, entry) -> str:
        """
        Extract content from RSS entry.
        
        Args:
            entry: RSS entry from feedparser
            
        Returns:
            Cleaned content text
        """
        content = ""
        
        # Try different content fields
        content_fields = ['content', 'summary', 'description']
        
        for field in content_fields:
            if hasattr(entry, field):
                field_content = getattr(entry, field)
                
                if isinstance(field_content, list) and field_content:
                    # Handle content list (e.g., from content field)
                    content = field_content[0].get('value', '')
                elif isinstance(field_content, str):
                    content = field_content
                
                if content:
                    break
        
        if not content:
            return ""
        
        # Clean and process content
        content = self._clean_html(content) if self.config.clean_html else content
        content = self._clean_text(content) if self.config.normalize_text else content
        
        # Extract full content if enabled and content is too short
        if (self.config.extract_full_content and 
            len(content) < self.config.min_content_length and 
            hasattr(entry, 'link')):
            full_content = self._extract_full_content(entry.link)
            if full_content and len(full_content) > len(content):
                content = full_content
        
        return content
    
    def _extract_full_content(self, article_url: str) -> Optional[str]:
        """
        Extract full content from article URL.
        
        Args:
            article_url: URL of the full article
            
        Returns:
            Full article content or None if extraction fails
        """
        try:
            response = self.session.get(
                article_url, 
                timeout=self.config.content_timeout
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Try to find main content
            content_selectors = [
                'article',
                '.article-content',
                '.post-content',
                '.entry-content',
                '.content',
                'main',
                '.main-content'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = elements[0].get_text(strip=True)
                    break
            
            # Fallback to body content
            if not content:
                body = soup.find('body')
                if body:
                    content = body.get_text(strip=True)
            
            # Clean and validate content
            if content:
                content = self._clean_text(content)
                if len(content) >= self.config.min_content_length:
                    return content
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract full content from {article_url}: {e}")
            return None
    
    def _extract_date(self, entry) -> Optional[datetime]:
        """
        Extract publication date from RSS entry.
        
        Args:
            entry: RSS entry from feedparser
            
        Returns:
            Publication date or None if not found
        """
        # Try different date fields
        date_fields = ['published_parsed', 'updated_parsed', 'published', 'updated']
        
        for field in date_fields:
            if hasattr(entry, field):
                date_value = getattr(entry, field)
                
                if date_value:
                    try:
                        if isinstance(date_value, tuple):
                            # Handle time.struct_time
                            return datetime(*date_value[:6], tzinfo=timezone.utc)
                        elif isinstance(date_value, str):
                            # Parse date string
                            return date_parser.parse(date_value)
                    except Exception as e:
                        logger.debug(f"Error parsing date {date_value}: {e}")
                        continue
        
        return None
    
    def _extract_author(self, entry) -> Optional[str]:
        """
        Extract author from RSS entry.
        
        Args:
            entry: RSS entry from feedparser
            
        Returns:
            Author name or None if not found
        """
        author_fields = ['author', 'dc_creator']
        
        for field in author_fields:
            if hasattr(entry, field):
                author = getattr(entry, field)
                if author and isinstance(author, str):
                    return self._clean_text(author)
        
        return None
    
    def _clean_html(self, content: str) -> str:
        """
        Clean HTML tags and entities from content.
        
        Args:
            content: Raw HTML content
            
        Returns:
            Cleaned text content
        """
        if not content:
            return ""
        
        # Parse HTML
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'img', 'video', 'audio']):
            element.decompose()
        
        # Get text content
        text = soup.get_text(separator=' ', strip=True)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-"\']', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.,!?;:]{2,}', '.', text)
        
        # Trim and return
        return text.strip()
    
    def _categorize_article(self, article: ParsedArticle) -> None:
        """
        Categorize article by sport and league.
        
        Args:
            article: Article to categorize
        """
        content_lower = f"{article.title} {article.content}".lower()
        
        # Detect sport with improved logic
        sport_scores = {}
        
        for sport, keywords in self.config.sport_keywords.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in content_lower:
                     # Give higher scores for more specific keywords
                     if sport == 'basketball' and keyword in ['basketball', 'nba', 'euroleague basketball']:
                         score += 15
                     elif sport == 'basketball' and keyword in ['euroleague', 'fiba', 'wnba']:
                         score += 10
                     elif sport == 'tennis' and keyword in ['tennis', 'wimbledon']:
                         score += 15
                     elif sport == 'cricket' and keyword in ['cricket', 'ipl']:
                         score += 15
                     elif sport == 'rugby' and keyword in ['rugby']:
                         score += 15
                     elif sport == 'golf' and keyword in ['golf', 'pga']:
                         score += 15
                     elif sport == 'football':
                         # Give higher scores for specific football terms
                         if keyword in ['football', 'soccer', 'premier league', 'la liga', 'bundesliga', 'serie a', 'ligue 1', 'uefa champions league', 'fifa world cup']:
                             score += 5
                         else:
                             score += 1
                     else:
                         score += 1
                     
                     matched_keywords.append(keyword)
            
            if score > 0:
                sport_scores[sport] = score
        
        # Assign sport based on highest score
        if sport_scores:
            article.sport = max(sport_scores, key=sport_scores.get)
        else:
            article.sport = ""
        
        # Detect specific leagues/competitions
        league_patterns = {
            'premier_league': ['premier league', 'epl'],
            'championship': ['championship', 'sky bet championship'],
            'nba': ['nba', 'national basketball association'],
            'nfl': ['nfl', 'national football league'],
            'uefa': ['uefa', 'champions league', 'europa league'],
            'fifa': ['fifa', 'world cup'],
            'wimbledon': ['wimbledon'],
            'ipl': ['ipl', 'indian premier league']
        }
        
        for league, patterns in league_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                article.league = league
                break
        
        # Extract tags
        article.tags = self._extract_tags(content_lower)
    
    def _extract_tags(self, content: str) -> List[str]:
        """
        Extract relevant tags from content.
        
        Args:
            content: Article content (lowercase)
            
        Returns:
            List of extracted tags
        """
        tags = []
        
        # Common sports terms
        sports_terms = [
            'transfer', 'goal', 'match', 'game', 'season', 'player', 'team',
            'coach', 'manager', 'injury', 'win', 'loss', 'draw', 'score',
            'championship', 'tournament', 'league', 'cup', 'final', 'semi-final'
        ]
        
        for term in sports_terms:
            if term in content:
                tags.append(term)
        
        return tags[:10]  # Limit to 10 tags
    
    def _calculate_quality_score(self, article: ParsedArticle) -> float:
        """
        Calculate quality score for an article.
        
        Args:
            article: Article to score
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        
        # Title quality (20%)
        if article.title and len(article.title) > 10:
            score += 0.2
        
        # Content length (30%)
        if article.word_count >= self.config.min_content_length:
            if article.word_count >= 100:
                score += 0.3
            else:
                score += 0.15
        
        # Has publication date (10%)
        if article.published_date:
            score += 0.1
        
        # Has author (10%)
        if article.author:
            score += 0.1
        
        # Sport categorization (15%)
        if article.sport:
            score += 0.15
        
        # Content quality indicators (15%)
        content_lower = article.content.lower()
        
        # Check for spam/promotional content
        spam_indicators = ['click here', 'subscribe', 'advertisement', 'sponsored']
        if any(indicator in content_lower for indicator in spam_indicators):
            score -= 0.2
        
        # Check for sports-specific content
        if any(keyword in content_lower for keywords in self.config.sport_keywords.values() for keyword in keywords):
            score += 0.15
        
        return max(0.0, min(1.0, score))
    
    def _is_valid_article(self, article: ParsedArticle) -> bool:
        """
        Validate if article meets quality criteria.
        
        Args:
            article: Article to validate
            
        Returns:
            True if article is valid
        """
        # Check minimum content length
        if len(article.content) < self.config.min_content_length:
            return False
        
        # Check maximum content length
        if len(article.content) > self.config.max_content_length:
            return False
        
        # Check for duplicate content
        if self.config.filter_duplicates:
            if article.content_hash in self.seen_hashes:
                return False
            self.seen_hashes.add(article.content_hash)
        
        # Check for excluded patterns
        content_lower = article.content.lower()
        for pattern in self.config.exclude_patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                return False
        
        # Check minimum quality score
        if article.quality_score < 0.3:
            return False
        
        return True
    
    def filter_articles_by_date(self, articles: List[ParsedArticle], 
                               hours_back: int = 24) -> List[ParsedArticle]:
        """
        Filter articles by publication date.
        
        Args:
            articles: List of articles to filter
            hours_back: Number of hours back to include
            
        Returns:
            Filtered list of articles
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        filtered_articles = []
        for article in articles:
            if article.published_date and article.published_date >= cutoff_date:
                filtered_articles.append(article)
            elif not article.published_date:
                # Include articles without date (assume recent)
                filtered_articles.append(article)
        
        return filtered_articles
    
    def filter_articles_by_sport(self, articles: List[ParsedArticle], 
                                sport: str) -> List[ParsedArticle]:
        """
        Filter articles by sport.
        
        Args:
            articles: List of articles to filter
            sport: Sport to filter by
            
        Returns:
            Filtered list of articles
        """
        return [article for article in articles if article.sport.lower() == sport.lower()]
    
    def get_parsing_statistics(self) -> Dict[str, Any]:
        """
        Get parsing statistics.
        
        Returns:
            Dictionary containing parsing statistics
        """
        return {
            'seen_hashes_count': len(self.seen_hashes),
            'duplicate_filter_enabled': self.config.filter_duplicates,
            'min_content_length': self.config.min_content_length,
            'max_content_length': self.config.max_content_length,
            'extract_full_content': self.config.extract_full_content,
            'supported_sports': list(self.config.sport_keywords.keys())
        }
    
    def clear_duplicate_cache(self) -> None:
        """
        Clear the duplicate detection cache.
        """
        self.seen_hashes.clear()
        logger.info("Cleared duplicate detection cache")