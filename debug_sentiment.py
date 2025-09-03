#!/usr/bin/env python3
"""Debug script to understand sentiment analysis issues."""

import os
import sys
from datetime import datetime, timedelta
from sqlalchemy import text

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and initialize config first
from formfinder.config import load_config, get_config
load_config()

# Now import database modules
from formfinder.database import get_db_manager
from newsapi import NewsApiClient

# Initialize NewsAPI client with hardcoded key for testing
NEWS_API_KEY = "ff008e7b4e9b4041ab44c50a729d7885"
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

def test_newsapi_for_team(team_name, days_back=7):
    """Test NewsAPI for a specific team."""
    from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')
    
    query = f'"{team_name}" AND (football OR soccer)'
    
    print(f"\n🔍 Testing NewsAPI for: {team_name}")
    print(f"   Query: {query}")
    print(f"   Date range: {from_date} to {to_date}")
    
    try:
        articles = newsapi.get_everything(
            q=query,
            from_param=from_date,
            to=to_date,
            language='en',
            sort_by='relevancy',
            page_size=10
        )
        
        total_results = articles.get('totalResults', 0)
        articles_list = articles.get('articles', [])
        
        print(f"   ✅ Found {len(articles_list)} articles out of {total_results} total")
        
        if articles_list:
            print(f"   📰 Sample article: {articles_list[0]['title'][:100]}...")
        
        return len(articles_list), total_results
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0, 0

def debug_database():
    """Debug the database for teams and fixtures."""
    print("🔍 Debugging database...")
    
    try:
        db_manager = get_db_manager()
        session = db_manager.get_session()
        
        # Check teams
        teams_result = session.execute(text("SELECT id, name FROM teams ORDER BY name"))
        teams = teams_result.fetchall()
        print(f"\n📊 Found {len(teams)} teams:")
        for team_id, team_name in teams[:15]:  # Show first 15
            print(f"   {team_id}: {team_name}")
        if len(teams) > 15:
            print(f"   ... and {len(teams) - 15} more")
        
        # Check all fixtures (not just upcoming)
        fixtures_result = session.execute(
            text("""
                SELECT f.id, f.match_date, ht.name as home_team, at.name as away_team, f.status
                FROM fixtures f
                JOIN teams ht ON f.home_team_id = ht.id
                JOIN teams at ON f.away_team_id = at.id
                ORDER BY f.match_date DESC
                LIMIT 20
            """)
        )
        fixtures = fixtures_result.fetchall()
        
        print(f"\n📅 Found {len(fixtures)} recent fixtures (showing latest 20):")
        for fixture_id, match_date, home_team, away_team, status in fixtures:
            date_str = match_date.strftime('%Y-%m-%d') if match_date else "No date"
            print(f"   {date_str} - {home_team} vs {away_team} ({status})")
        
        # Check upcoming fixtures specifically
        today = datetime.now()
        upcoming_fixtures_result = session.execute(
            text("""
                SELECT f.id, f.match_date, ht.name as home_team, at.name as away_team
                FROM fixtures f
                JOIN teams ht ON f.home_team_id = ht.id
                JOIN teams at ON f.away_team_id = at.id
                WHERE f.match_date >= :today
                ORDER BY f.match_date ASC
                LIMIT 10
            """),
            {"today": today}
        )
        upcoming_fixtures = upcoming_fixtures_result.fetchall()
        
        print(f"\n📅 Found {len(upcoming_fixtures)} upcoming fixtures:")
        for fixture_id, match_date, home_team, away_team in upcoming_fixtures:
            date_str = match_date.strftime('%Y-%m-%d') if match_date else "No date"
            print(f"   {date_str} - {home_team} vs {away_team}")
        
        # Test NewsAPI for major teams regardless of fixtures
        print(f"\n🧪 Testing NewsAPI for major teams...")
        major_teams = [
            "Arsenal", "Chelsea", "Manchester United", "Liverpool", 
            "Manchester City", "Tottenham", "Barcelona", "Real Madrid",
            "Bayern Munich", "PSG", "Juventus", "AC Milan"
        ]
        
        successful_teams = []
        failed_teams = []
        
        for team in major_teams:
            articles, total = test_newsapi_for_team(team)
            if articles > 0:
                successful_teams.append((team, articles, total))
            else:
                failed_teams.append(team)
        
        print(f"\n📊 Summary:")
        print(f"   ✅ Successful: {len(successful_teams)} teams")
        print(f"   ❌ Failed: {len(failed_teams)} teams")
        
        if successful_teams:
            print(f"\n   Most successful teams:")
            for team, articles, total in sorted(successful_teams, key=lambda x: x[2], reverse=True)[:5]:
                print(f"      {team}: {articles} articles (of {total} total)")
        
        if failed_teams:
            print(f"\n   Failed teams: {', '.join(failed_teams)}")
        
        # Test a few smaller teams
        print(f"\n🧪 Testing smaller teams...")
        smaller_teams = ["Burnley", "Watford", "Norwich", "Sheffield United"]
        for team in smaller_teams:
            test_newsapi_for_team(team)
        
        session.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting comprehensive sentiment analysis debugging...")
    debug_database()