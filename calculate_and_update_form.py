#!/usr/bin/env python3
"""Calculate and update form column in standings table based on recent match results."""

from formfinder.config import load_config
from formfinder.database import get_db_session
from sqlalchemy import text
from datetime import datetime, timedelta

def calculate_team_form(session, team_id: int, league_id: int, num_games: int = 5) -> str:
    """Calculate team form based on recent match results.
    
    Args:
        session: Database session
        team_id: Team ID
        league_id: League ID
        num_games: Number of recent games to consider
        
    Returns:
        Form string like 'WWDLL' or empty string if no matches
    """
    # Get recent finished matches for the team
    query = text("""
        SELECT 
            CASE 
                WHEN (home_team_id = :team_id AND home_score > away_score) OR 
                     (away_team_id = :team_id AND away_score > home_score) THEN 'W'
                WHEN home_score = away_score THEN 'D'
                ELSE 'L'
            END as result
        FROM fixtures
        WHERE (home_team_id = :team_id OR away_team_id = :team_id)
            AND league_id = :league_id
            AND status = 'finished'
            AND home_score IS NOT NULL
            AND away_score IS NOT NULL
        ORDER BY match_date DESC
        LIMIT :num_games
    """)
    
    results = session.execute(query, {
        'team_id': team_id,
        'league_id': league_id,
        'num_games': num_games
    }).fetchall()
    
    # Build form string from most recent to oldest
    form_string = ''.join([result[0] for result in results])
    return form_string

def update_standings_form():
    """Update form column for all standings records."""
    load_config()
    
    with get_db_session() as session:
        print("Calculating and updating form for all standings...")
        
        # Get all standings records
        standings_query = text("""
            SELECT id, team_id, league_id
            FROM standings
            ORDER BY league_id, team_id
        """)
        
        standings = session.execute(standings_query).fetchall()
        print(f"Found {len(standings)} standings records to update")
        
        updated_count = 0
        
        for standing in standings:
            standing_id, team_id, league_id = standing
            
            # Calculate form for this team
            form = calculate_team_form(session, team_id, league_id)
            
            if form:  # Only update if we have form data
                # Update the form column
                update_query = text("""
                    UPDATE standings 
                    SET form = :form, updated_at = :updated_at
                    WHERE id = :standing_id
                """)
                
                session.execute(update_query, {
                    'form': form,
                    'updated_at': datetime.utcnow(),
                    'standing_id': standing_id
                })
                
                updated_count += 1
                
                if updated_count % 50 == 0:
                    print(f"Updated {updated_count} standings...")
        
        # Commit all changes
        session.commit()
        
        print(f"\nForm update complete!")
        print(f"Total standings updated: {updated_count}")
        
        # Show some sample results
        print("\nSample standings with form data:")
        sample_query = text("""
            SELECT team_id, form
            FROM standings
            WHERE form IS NOT NULL AND form != ''
            LIMIT 10
        """)
        
        samples = session.execute(sample_query).fetchall()
        for team_id, form in samples:
            print(f"Team {team_id}: {form}")
        
        # Check statistics
        stats_query = text("""
            SELECT 
                COUNT(*) as total_standings,
                COUNT(CASE WHEN form IS NOT NULL AND form != '' THEN 1 END) as with_form,
                COUNT(CASE WHEN form IS NULL OR form = '' THEN 1 END) as without_form
            FROM standings
        """)
        
        stats = session.execute(stats_query).fetchone()
        print(f"\nForm statistics:")
        print(f"Total standings: {stats[0]}")
        print(f"With form data: {stats[1]}")
        print(f"Without form data: {stats[2]}")

if __name__ == '__main__':
    update_standings_form()