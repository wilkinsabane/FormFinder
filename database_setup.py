import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.sql import func
from datetime import datetime

DATABASE_URL = "sqlite:///formfinder.db"
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class League(Base):
    __tablename__ = "leagues"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    api_league_id = Column(Integer, unique=True, index=True, nullable=False) # Store the ID from the API

    matches = relationship("Match", back_populates="league")
    standings = relationship("Standing", back_populates="league")
    high_form_teams = relationship("HighFormTeam", back_populates="league")

class Team(Base):
    __tablename__ = "teams"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    api_team_id = Column(String, unique=True, index=True, nullable=False) # Store the ID from the API

    home_matches = relationship("Match", foreign_keys="[Match.home_team_id]", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="[Match.away_team_id]", back_populates="away_team")
    standings = relationship("Standing", back_populates="team")
    high_form_entries = relationship("HighFormTeam", back_populates="team")


class Match(Base):
    __tablename__ = "matches"
    id = Column(Integer, primary_key=True, index=True)
    api_match_id = Column(String, unique=True, index=True, nullable=False) # From API
    date = Column(DateTime, nullable=False)
    time = Column(String, nullable=True) # Store as string HH:MM

    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=False)
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)

    status = Column(String, nullable=True) # e.g., 'finished', 'scheduled'
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)

    # For upcoming fixtures, scores will be null
    is_fixture = Column(Integer, default=0) # 0 for historical, 1 for fixture

    league = relationship("League", back_populates="matches")
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")

    __table_args__ = (
        Index("idx_match_league_date", "league_id", "date"),
        Index("idx_match_home_team_date", "home_team_id", "date"),
        Index("idx_match_away_team_date", "away_team_id", "date"),
    )

class Standing(Base):
    __tablename__ = "standings"
    id = Column(Integer, primary_key=True, index=True)

    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    season_year = Column(String, nullable=False) # e.g., "2024-2025"

    position = Column(Integer)
    games_played = Column(Integer)
    points = Column(Integer)
    wins = Column(Integer)
    draws = Column(Integer)
    losses = Column(Integer)
    goals_for = Column(Integer)
    goals_against = Column(Integer)
    fetched_at = Column(DateTime, default=func.now())

    league = relationship("League", back_populates="standings")
    team = relationship("Team", back_populates="standings")

    __table_args__ = (
        UniqueConstraint("league_id", "team_id", "season_year", name="uq_standing_league_team_season"),
        Index("idx_standing_league_team", "league_id", "team_id"),
    )

class HighFormTeam(Base):
    __tablename__ = "high_form_teams"
    id = Column(Integer, primary_key=True, index=True)
    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    win_rate = Column(Float, nullable=False)
    recent_period_games = Column(Integer, nullable=False) # Number of games considered for form
    calculated_at = Column(DateTime, default=func.now())
    season_year = Column(String, nullable=False) # e.g., "2024-2025", to associate with specific historical data

    league = relationship("League", back_populates="high_form_teams")
    team = relationship("Team", back_populates="high_form_entries")

    __table_args__ = (
        UniqueConstraint("league_id", "team_id", "season_year", "recent_period_games", name="uq_high_form_team_league_season_period"),
         Index("idx_high_form_team_league_team", "league_id", "team_id"),
    )


def create_db_and_tables():
    # This will create the database file if it doesn't exist
    # and create all tables defined that inherit from Base
    Base.metadata.create_all(bind=engine)
    print("Database and tables created successfully (if they didn't exist).")

if __name__ == "__main__":
    create_db_and_tables()
    # Example: Add a league
    # db = SessionLocal()
    # try:
    #     new_league = League(name="Example Premier League", api_league_id=123)
    #     db.add(new_league)
    #     db.commit()
    #     db.refresh(new_league)
    #     print(f"Added league: {new_league.name} with id {new_league.id}")
    # finally:
    #     db.close()
