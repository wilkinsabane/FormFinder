"""Database-based H2H statistics computation.

This module computes head-to-head statistics directly from the fixtures database
instead of relying on API calls, making feature precomputation more efficient.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session


class DatabaseH2HComputer:
    """Computes H2H statistics directly from the fixtures database."""
    
    def __init__(self, db_session: Session):
        """Initialize the H2H computer with database session.
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
        
    def compute_h2h_stats(self, team1_id: int, team2_id: int, 
                         competition_id: Optional[int] = None,
                         max_matches: int = 10) -> Dict[str, Any]:
        """Compute H2H statistics from database fixtures.
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID  
            competition_id: Optional competition filter
            max_matches: Maximum number of historical matches to consider
            
        Returns:
            Dictionary containing H2H statistics
        """
        try:
            # Build query to get historical matches between teams
            query_conditions = [
                "((f.home_team_id = :team1_id AND f.away_team_id = :team2_id) OR",
                " (f.home_team_id = :team2_id AND f.away_team_id = :team1_id))",
                "AND f.status = 'finished'",
                "AND f.home_score IS NOT NULL",
                "AND f.away_score IS NOT NULL"
            ]
            
            params = {
                'team1_id': team1_id,
                'team2_id': team2_id,
                'max_matches': max_matches
            }
            
            if competition_id:
                query_conditions.append("AND f.league_id = :competition_id")
                params['competition_id'] = competition_id
                
            query = text(f"""
                SELECT 
                    f.home_team_id,
                    f.away_team_id,
                    f.home_score,
                    f.away_score,
                    f.match_date,
                    f.league_id
                FROM fixtures f
                WHERE {' '.join(query_conditions)}
                ORDER BY f.match_date DESC
                LIMIT :max_matches
            """)
            
            result = self.db_session.execute(query, params)
            matches = result.fetchall()
            
            if not matches:
                self.logger.debug(f"No H2H matches found for teams {team1_id} vs {team2_id}")
                return self._get_default_h2h_stats(team1_id, team2_id, competition_id)
                
            # Compute statistics from matches
            return self._compute_stats_from_matches(matches, team1_id, team2_id, competition_id)
            
        except Exception as e:
            self.logger.error(f"Error computing H2H stats from database: {e}")
            return self._get_default_h2h_stats(team1_id, team2_id, competition_id)
            
    def _compute_stats_from_matches(self, matches, team1_id: int, team2_id: int, 
                                  competition_id: Optional[int]) -> Dict[str, Any]:
        """Compute statistics from match results.
        
        Args:
            matches: List of match results from database
            team1_id: First team ID
            team2_id: Second team ID
            competition_id: Competition ID
            
        Returns:
            Dictionary with computed H2H statistics
        """
        total_matches = len(matches)
        team1_wins = 0
        team2_wins = 0
        draws = 0
        team1_goals = 0
        team2_goals = 0
        
        # Home/away specific stats
        team1_home_games = 0
        team1_home_wins = 0
        team1_home_draws = 0
        team1_home_losses = 0
        team1_home_scored = 0
        team1_home_conceded = 0
        
        team2_home_games = 0
        team2_home_wins = 0
        team2_home_draws = 0
        team2_home_losses = 0
        team2_home_scored = 0
        team2_home_conceded = 0
        
        for match in matches:
            home_team_id = match.home_team_id
            away_team_id = match.away_team_id
            home_score = match.home_score
            away_score = match.away_score
            
            # Determine which team was home/away
            if home_team_id == team1_id:
                # Team1 was home, Team2 was away
                team1_goals += home_score
                team2_goals += away_score
                
                team1_home_games += 1
                team1_home_scored += home_score
                team1_home_conceded += away_score
                
                if home_score > away_score:
                    team1_wins += 1
                    team1_home_wins += 1
                elif home_score < away_score:
                    team2_wins += 1
                    team1_home_losses += 1
                else:
                    draws += 1
                    team1_home_draws += 1
                    
            else:
                # Team2 was home, Team1 was away
                team1_goals += away_score
                team2_goals += home_score
                
                team2_home_games += 1
                team2_home_scored += home_score
                team2_home_conceded += away_score
                
                if away_score > home_score:
                    team1_wins += 1
                    team2_home_losses += 1
                elif away_score < home_score:
                    team2_wins += 1
                    team2_home_wins += 1
                else:
                    draws += 1
                    team2_home_draws += 1
                    
        # Calculate averages
        avg_total_goals = (team1_goals + team2_goals) / total_matches if total_matches > 0 else 0.0
        
        return {
            # Overall statistics
            'overall_games_played': total_matches,
            'overall_team1_wins': team1_wins,
            'overall_team2_wins': team2_wins,
            'overall_draws': draws,
            'overall_team1_scored': team1_goals,
            'overall_team2_scored': team2_goals,
            'avg_total_goals': round(avg_total_goals, 2),
            
            # Home/away breakdowns
            'team1_games_played_at_home': team1_home_games,
            'team1_wins_at_home': team1_home_wins,
            'team1_draws_at_home': team1_home_draws,
            'team1_losses_at_home': team1_home_losses,
            'team1_scored_at_home': team1_home_scored,
            'team1_conceded_at_home': team1_home_conceded,
            
            'team2_games_played_at_home': team2_home_games,
            'team2_wins_at_home': team2_home_wins,
            'team2_draws_at_home': team2_home_draws,
            'team2_losses_at_home': team2_home_losses,
            'team2_scored_at_home': team2_home_scored,
            'team2_conceded_at_home': team2_home_conceded,
            
            # Metadata
            'team1_id': team1_id,
            'team2_id': team2_id,
            'competition_id': competition_id,
            'computed_at': datetime.now(),
            'source': 'database'
        }
        
    def _get_default_h2h_stats(self, team1_id: int, team2_id: int, 
                              competition_id: Optional[int]) -> Dict[str, Any]:
        """Return default H2H statistics when no data is available.
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            competition_id: Competition ID
            
        Returns:
            Dictionary with default H2H statistics
        """
        return {
            # Overall statistics
            'overall_games_played': 0,
            'overall_team1_wins': 0,
            'overall_team2_wins': 0,
            'overall_draws': 0,
            'overall_team1_scored': 0,
            'overall_team2_scored': 0,
            'avg_total_goals': 0.0,
            
            # Home/away breakdowns
            'team1_games_played_at_home': 0,
            'team1_wins_at_home': 0,
            'team1_draws_at_home': 0,
            'team1_losses_at_home': 0,
            'team1_scored_at_home': 0,
            'team1_conceded_at_home': 0,
            
            'team2_games_played_at_home': 0,
            'team2_wins_at_home': 0,
            'team2_draws_at_home': 0,
            'team2_losses_at_home': 0,
            'team2_scored_at_home': 0,
            'team2_conceded_at_home': 0,
            
            # Metadata
            'team1_id': team1_id,
            'team2_id': team2_id,
            'competition_id': competition_id,
            'computed_at': datetime.now(),
            'source': 'database'
        }
        
    def cache_h2h_stats(self, team1_id: int, team2_id: int, 
                       competition_id: Optional[int], h2h_data: Dict[str, Any]):
        """Cache H2H statistics in the h2h_cache_enhanced table.
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            competition_id: Competition ID
            h2h_data: H2H statistics to cache
        """
        try:
            # Delete existing cache entry
            delete_query = text("""
                DELETE FROM h2h_cache_enhanced 
                WHERE ((team1_id = :team1_id AND team2_id = :team2_id) OR 
                       (team1_id = :team2_id AND team2_id = :team1_id))
                AND COALESCE(competition_id, 0) = COALESCE(:competition_id, 0)
            """)
            
            self.db_session.execute(delete_query, {
                'team1_id': team1_id,
                'team2_id': team2_id,
                'competition_id': competition_id
            })
            
            # Insert new cache entry
            insert_query = text("""
                INSERT INTO h2h_cache_enhanced (
                    team1_id, team2_id, competition_id,
                    total_games, team1_wins, team2_wins, draws,
                    total_goals_scored, avg_total_goals, avg_team1_goals, avg_team2_goals,
                    team1_home_games, team1_home_wins, team1_home_draws, team1_home_losses,
                    team1_home_goals_for, team1_home_goals_against,
                    team2_home_games, team2_home_wins, team2_home_draws, team2_home_losses,
                    team2_home_goals_for, team2_home_goals_against,
                    last_updated, cache_expires_at
                ) VALUES (
                    :team1_id, :team2_id, :competition_id,
                    :total_games, :team1_wins, :team2_wins, :draws,
                    :total_goals_scored, :avg_total_goals, :avg_team1_goals, :avg_team2_goals,
                    :team1_home_games, :team1_home_wins, :team1_home_draws, :team1_home_losses,
                    :team1_home_goals_for, :team1_home_goals_against,
                    :team2_home_games, :team2_home_wins, :team2_home_draws, :team2_home_losses,
                    :team2_home_goals_for, :team2_home_goals_against,
                    NOW(), NOW() + INTERVAL '24 hours'
                )
            """)
            
            self.db_session.execute(insert_query, {
                'team1_id': team1_id,
                'team2_id': team2_id,
                'competition_id': competition_id,
                'total_games': h2h_data['overall_games_played'],
                'team1_wins': h2h_data['overall_team1_wins'],
                'team2_wins': h2h_data['overall_team2_wins'],
                'draws': h2h_data['overall_draws'],
                'total_goals_scored': h2h_data['overall_team1_scored'] + h2h_data['overall_team2_scored'],
                'avg_total_goals': h2h_data['avg_total_goals'],
                'avg_team1_goals': h2h_data['overall_team1_scored'] / max(h2h_data['overall_games_played'], 1),
                'avg_team2_goals': h2h_data['overall_team2_scored'] / max(h2h_data['overall_games_played'], 1),
                'team1_home_games': h2h_data['team1_games_played_at_home'],
                'team1_home_wins': h2h_data['team1_wins_at_home'],
                'team1_home_draws': h2h_data['team1_draws_at_home'],
                'team1_home_losses': h2h_data['team1_losses_at_home'],
                'team1_home_goals_for': h2h_data['team1_scored_at_home'],
                'team1_home_goals_against': h2h_data['team1_conceded_at_home'],
                'team2_home_games': h2h_data['team2_games_played_at_home'],
                'team2_home_wins': h2h_data['team2_wins_at_home'],
                'team2_home_draws': h2h_data['team2_draws_at_home'],
                'team2_home_losses': h2h_data['team2_losses_at_home'],
                'team2_home_goals_for': h2h_data['team2_scored_at_home'],
                'team2_home_goals_against': h2h_data['team2_conceded_at_home']
            })
            
            self.db_session.commit()
            self.logger.debug(f"Cached H2H data for teams {team1_id} vs {team2_id}")
            
        except Exception as e:
            self.logger.error(f"Error caching H2H data: {e}")
            self.db_session.rollback()
            
    def get_cached_h2h_stats(self, team1_id: int, team2_id: int, 
                            competition_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Retrieve cached H2H statistics from database.
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            competition_id: Optional competition filter
            
        Returns:
            Cached H2H statistics or None if not found/expired
        """
        try:
            query = text("""
                SELECT *
                FROM h2h_cache_enhanced 
                WHERE ((team1_id = :team1_id AND team2_id = :team2_id) OR 
                       (team1_id = :team2_id AND team2_id = :team1_id))
                AND COALESCE(competition_id, 0) = COALESCE(:competition_id, 0)
                AND cache_expires_at > NOW()
                ORDER BY last_updated DESC
                LIMIT 1
            """)
            
            result = self.db_session.execute(query, {
                'team1_id': team1_id,
                'team2_id': team2_id,
                'competition_id': competition_id
            }).fetchone()
            
            if result:
                return dict(result._mapping)
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving cached H2H data: {e}")
            return None