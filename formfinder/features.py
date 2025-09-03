from __future__ import annotations
from typing import Dict, Any, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session
from datetime import datetime
from .exceptions import FeatureError
from .clients.api_client import SoccerDataAPIClient

# ---------- Rolling form (DB-driven) ----------
def get_rolling_form_features(
    team_id: int,
    match_date: datetime,
    db_session: Session,
    last_n_games: int = 5
) -> Dict[str, float]:
    """
    Calculates rolling averages for goals scored/conceded overall and by home/away.
    Assumes fixtures table with columns:
      fixture_id, match_date, status, home_team_id, away_team_id, home_score, away_score
    """
    q = text("""
        WITH recent AS (
          SELECT
            match_date,
            (home_team_id = :team_id) AS is_home,
            CASE WHEN home_team_id = :team_id THEN home_score ELSE away_score END AS goals_scored,
            CASE WHEN home_team_id = :team_id THEN away_score ELSE home_score END AS goals_conceded
          FROM fixtures
          WHERE (home_team_id = :team_id OR away_team_id = :team_id)
            AND match_date < :match_date
            AND status = 'finished'
          ORDER BY match_date DESC
          LIMIT :limit
        )
        SELECT
          AVG(goals_scored)::float AS avg_goals_scored,
          AVG(goals_conceded)::float AS avg_goals_conceded,
          AVG(CASE WHEN is_home THEN goals_scored END)::float AS avg_goals_scored_home,
          AVG(CASE WHEN NOT is_home THEN goals_scored END)::float AS avg_goals_scored_away,
          AVG(CASE WHEN is_home THEN goals_conceded END)::float AS avg_goals_conceded_home,
          AVG(CASE WHEN NOT is_home THEN goals_conceded END)::float AS avg_goals_conceded_away
        FROM recent;
    """)
    try:
        row = db_session.execute(q, {"team_id": team_id, "match_date": match_date, "limit": last_n_games}).mappings().first()
    except Exception as e:
        # Rollback the transaction and re-raise with more context
        db_session.rollback()
        raise FeatureError(f"Failed to fetch rolling form for team {team_id}: {e}")
    
    # Default 0.0 if no history
    def f(k): return float(row[k]) if row and row[k] is not None else 0.0
    return {
        "avg_goals_scored_last_5": f("avg_goals_scored"),
        "avg_goals_conceded_last_5": f("avg_goals_conceded"),
        "avg_goals_scored_home_last_5": f("avg_goals_scored_home"),
        "avg_goals_scored_away_last_5": f("avg_goals_scored_away"),
        "avg_goals_conceded_home_last_5": f("avg_goals_conceded_home"),
        "avg_goals_conceded_away_last_5": f("avg_goals_conceded_away"),
    }

# ---------- H2H (API-driven, DB cached by client) ----------
def get_h2h_feature(home_team_id: int, away_team_id: int, api_client: SoccerDataAPIClient, competition_id: Optional[int] = None) -> Dict[str, Any]:
    d = api_client.get_h2h_stats(home_team_id, away_team_id, competition_id=competition_id)
    # Standardize output naming for training
    return {
        "h2h_overall_games": d["overall_games_played"],
        "h2h_overall_team1_goals": d["overall_team1_scored"],
        "h2h_overall_team2_goals": d["overall_team2_scored"],
        "h2h_avg_total_goals": d["avg_total_goals"],
        # Optional advanced splits:
        "h2h_team1_games_played_at_home": d.get("team1_games_played_at_home"),
        "h2h_team1_wins_at_home": d.get("team1_wins_at_home"),
        "h2h_team1_losses_at_home": d.get("team1_losses_at_home"),
        "h2h_team1_draws_at_home": d.get("team1_draws_at_home"),
        "h2h_team1_scored_at_home": d.get("team1_scored_at_home"),
        "h2h_team1_conceded_at_home": d.get("team1_conceded_at_home"),
        "h2h_team2_games_played_at_home": d.get("team2_games_played_at_home"),
        "h2h_team2_wins_at_home": d.get("team2_wins_at_home"),
        "h2h_team2_losses_at_home": d.get("team2_losses_at_home"),
        "h2h_team2_draws_at_home": d.get("team2_draws_at_home"),
        "h2h_team2_scored_at_home": d.get("team2_scored_at_home"),
        "h2h_team2_conceded_at_home": d.get("team2_conceded_at_home"),
    }

# ---------- Match Preview (API-driven) ----------
def get_preview_metrics(match_id: int, api_client: SoccerDataAPIClient, compute_sentiment: bool = False) -> Dict[str, float]:
    data = api_client.get_match_preview(match_id)
    md = data.get("match_data", {}) or {}
    if "excitement_rating" not in md:
        raise FeatureError("excitement_rating missing in match preview response")
    result = {"preview_excitement_rating": float(md["excitement_rating"])}

    # Extract weather information if available
    weather = md.get("weather", {})
    if weather:
        result["weather_temp_f"] = float(weather.get("temp_f", 0.0))
        result["weather_temp_c"] = float(weather.get("temp_c", 0.0))
        # Convert description to numerical feature (0=other, 1=sunny, 2=cloudy, 3=rainy)
        description = weather.get("description", "").lower()
        if "sunny" in description or "clear" in description:
            result["weather_condition"] = 1.0
        elif "cloud" in description or "overcast" in description:
            result["weather_condition"] = 2.0
        elif "rain" in description or "drizzle" in description or "shower" in description:
            result["weather_condition"] = 3.0
        else:
            result["weather_condition"] = 0.0
    else:
        # Default weather values when not available
        result["weather_temp_f"] = 70.0  # Default moderate temperature
        result["weather_temp_c"] = 21.0
        result["weather_condition"] = 0.0

    if compute_sentiment:
        try:
            content = " ".join([c.get("content", "") for c in (data.get("content") or [])])
            # plug-in your sentiment lib here; placeholder zero
            from textblob import TextBlob
            sentiment = TextBlob(content).sentiment.polarity if content else 0.0
            result["preview_sentiment"] = float(sentiment)
        except Exception:
            result["preview_sentiment"] = 0.0

    return result