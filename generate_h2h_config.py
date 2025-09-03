#!/usr/bin/env python3
"""
Automatic H2H Teams Configuration Generator

This script reads league IDs from free_leagues.txt and generates
a comprehensive h2h_teams_config.json file by querying the database
for teams and fixtures in those leagues.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from formfinder.database import get_db_session
from formfinder.config import get_config, load_config
from sqlalchemy import text


class H2HConfigGenerator:
    """Generates H2H team configuration from database and league files."""
    
    def __init__(self):
        # Load configuration if not already loaded
        try:
            self.config = get_config()
        except RuntimeError:
            load_config()
            self.config = get_config()
        self.league_names = {
            203: "Premier League",
            204: "La Liga", 
            205: "Serie A",
            206: "Ligue 1",
            207: "Bundesliga",
            208: "Primeira Liga",
            209: "Eredivisie",
            210: "Jupiler Pro League",
            211: "Super League",
            212: "Premier League",
            213: "Superliga",
            214: "Liga MX",
            215: "Brasileirão",
            216: "Primera División",
            # Add more as needed
        }
    
    def read_league_ids(self, file_path: str) -> List[int]:
        """Read league IDs from the free_leagues.txt file."""
        try:
            with open(file_path, 'r') as f:
                league_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
            print(f"📋 Found {len(league_ids)} league IDs in {file_path}")
            return league_ids
        except FileNotFoundError:
            print(f"❌ File {file_path} not found")
            return []
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return []
    
    def get_teams_by_league(self, session, league_id: int) -> List[Tuple[int, str]]:
        """Get teams that have played in a specific league."""
        query = text("""
            SELECT DISTINCT t.id, t.name
            FROM teams t
            JOIN fixtures f ON (f.home_team_id = t.id OR f.away_team_id = t.id)
            WHERE f.league_id = :league_id
            AND t.name IS NOT NULL
            ORDER BY t.name
            LIMIT 20
        """)
        
        result = session.execute(query, {"league_id": league_id})
        teams = result.fetchall()
        return [(team.id, team.name) for team in teams]
    
    def get_recent_fixtures(self, session, league_id: int) -> List[Tuple[int, int, str]]:
        """Get recent fixtures for a league to identify active team pairs."""
        query = text("""
            SELECT f.home_team_id, f.away_team_id,
                   ht.name || ' vs ' || at.name as matchup
            FROM fixtures f
            JOIN teams ht ON f.home_team_id = ht.id
            JOIN teams at ON f.away_team_id = at.id
            WHERE f.league_id = :league_id
            AND f.match_date >= CURRENT_DATE - INTERVAL '365 days'
            ORDER BY f.match_date DESC
            LIMIT 15
        """)
        
        result = session.execute(query, {"league_id": league_id})
        fixtures = result.fetchall()
        return [(fix.home_team_id, fix.away_team_id, fix.matchup) for fix in fixtures]
    
    def get_league_info(self, session, league_id: int) -> Optional[str]:
        """Get league name from database or use predefined mapping."""
        if league_id in self.league_names:
            return self.league_names[league_id]
        
        # Try to get from database
        query = text("SELECT name FROM leagues WHERE id = :league_id")
        result = session.execute(query, {"league_id": league_id})
        league = result.fetchone()
        
        if league and league.name:
            return league.name
        
        return f"League {league_id}"
    
    def generate_team_pairs(self, teams: List[Tuple[int, str]], 
                          fixtures: List[Tuple[int, int, str]]) -> List[Dict]:
        """Generate team pairs with priorities based on recent fixtures and team quality."""
        team_pairs = []
        used_pairs = set()
        
        # High priority: Recent fixture pairs
        for home_id, away_id, matchup in fixtures[:5]:
            pair_key = tuple(sorted([home_id, away_id]))
            if pair_key not in used_pairs:
                team_pairs.append({
                    "team1_id": home_id,
                    "team2_id": away_id,
                    "priority": "HIGH",
                    "description": f"Recent fixture: {matchup}"
                })
                used_pairs.add(pair_key)
        
        # Medium priority: More recent fixtures
        for home_id, away_id, matchup in fixtures[5:10]:
            pair_key = tuple(sorted([home_id, away_id]))
            if pair_key not in used_pairs:
                team_pairs.append({
                    "team1_id": home_id,
                    "team2_id": away_id,
                    "priority": "MEDIUM",
                    "description": f"Active teams: {matchup}"
                })
                used_pairs.add(pair_key)
        
        # Low priority: Additional team combinations
        if len(teams) >= 4:
            for i in range(0, min(len(teams), 8), 2):
                if i + 1 < len(teams):
                    team1_id, team1_name = teams[i]
                    team2_id, team2_name = teams[i + 1]
                    pair_key = tuple(sorted([team1_id, team2_id]))
                    
                    if pair_key not in used_pairs:
                        team_pairs.append({
                            "team1_id": team1_id,
                            "team2_id": team2_id,
                            "priority": "LOW",
                            "description": f"{team1_name} vs {team2_name}"
                        })
                        used_pairs.add(pair_key)
        
        return team_pairs[:10]  # Limit to 10 pairs per league
    
    async def generate_config(self, league_ids: List[int]) -> Dict:
        """Generate the complete H2H configuration."""
        config = {"leagues": {}}
        
        with get_db_session() as session:
            for league_id in league_ids:
                print(f"🔍 Processing league {league_id}...")
                
                # Get league info
                league_name = self.get_league_info(session, league_id)
                
                # Get teams and fixtures
                teams = self.get_teams_by_league(session, league_id)
                fixtures = self.get_recent_fixtures(session, league_id)
                
                if not teams:
                    print(f"⚠️  No teams found for league {league_id}")
                    continue
                
                if not fixtures:
                    print(f"⚠️  No recent fixtures found for league {league_id}")
                    # Still create pairs from available teams
                    fixtures = []
                
                # Generate team pairs
                team_pairs = self.generate_team_pairs(teams, fixtures)
                
                if team_pairs:
                    config["leagues"][str(league_id)] = {
                        "name": league_name,
                        "team_pairs": team_pairs
                    }
                    print(f"✅ Added {len(team_pairs)} team pairs for {league_name}")
                else:
                    print(f"⚠️  No valid team pairs generated for league {league_id}")
        
        return config
    
    async def save_config(self, config: Dict, output_file: str):
        """Save the configuration to a JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"💾 Configuration saved to {output_file}")
            
            # Print summary
            total_leagues = len(config["leagues"])
            total_pairs = sum(len(league["team_pairs"]) for league in config["leagues"].values())
            print(f"📊 Summary: {total_leagues} leagues, {total_pairs} team pairs")
            
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")


async def main():
    """Main function to generate H2H configuration."""
    print("🚀 Starting H2H Configuration Generator")
    
    # Initialize configuration
    try:
        load_config()
    except Exception as e:
        print(f"⚠️  Warning: Could not load config: {e}")
    
    generator = H2HConfigGenerator()
    
    # Read league IDs
    league_file = "free_leagues.txt"
    league_ids = generator.read_league_ids(league_file)
    
    if not league_ids:
        print("❌ No league IDs found. Exiting.")
        return
    
    # Limit to first 15 leagues to avoid overwhelming the system
    league_ids = league_ids[:15]
    print(f"🎯 Processing first {len(league_ids)} leagues")
    
    # Generate configuration
    config = await generator.generate_config(league_ids)
    
    if not config["leagues"]:
        print("❌ No valid leagues found. Exiting.")
        return
    
    # Save configuration
    output_file = "h2h_teams_config_auto.json"
    await generator.save_config(config, output_file)
    
    print("✅ H2H Configuration generation completed!")
    print(f"📁 Use: python formfinder/h2h_collection_service.py --config-file {output_file}")


if __name__ == "__main__":
    asyncio.run(main())