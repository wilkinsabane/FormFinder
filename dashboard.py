"""
Football Data Visualization Dashboard
A comprehensive Streamlit dashboard for visualizing football data from PostgreSQL database.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Football Data Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4788;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

class DatabaseManager:
    """Manages database connections and queries."""
    
    def __init__(self):
        self.engine = None
        self.connect()
        self._team_cache = None
        self._league_cache = None
    
    def connect(self):
        """Establish database connection."""
        try:
            # Try Streamlit secrets first, then environment variables
            if hasattr(st, 'secrets'):
                db_uri = st.secrets.get("DB_URI", os.getenv("DB_URI"))
            else:
                db_uri = os.getenv("DB_URI")
            
            if not db_uri:
                st.error("Database URI not found. Please set DB_URI environment variable or add to Streamlit secrets.")
                st.stop()
            
            self.engine = create_engine(db_uri)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection established successfully")
            
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            st.error(f"Failed to connect to database: {str(e)}")
            st.stop()
    
    @st.cache_data(ttl=600)
    def get_team_mappings(_self):
        """Get team ID to name mappings."""
        try:
            query = """
                SELECT id, name, short_name 
                FROM teams 
                ORDER BY name
            """
            return _self.fetch_data(query)
        except Exception as e:
            logger.error(f"Error loading team mappings: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=600)
    def get_league_mappings(_self):
        """Get league ID to name mappings."""
        try:
            query = """
                SELECT id, name, country, season 
                FROM leagues 
                ORDER BY name
            """
            return _self.fetch_data(query)
        except Exception as e:
            logger.error(f"Error loading league mappings: {str(e)}")
            return pd.DataFrame()
    
    def get_team_name(self, team_id):
        """Get team name from ID."""
        if self._team_cache is None:
            self._team_cache = self.get_team_mappings()
        
        if not self._team_cache.empty:
            team = self._team_cache[self._team_cache['id'] == team_id]
            if not team.empty:
                return team.iloc[0]['name']
        return f"Team {team_id}"
    
    def get_league_name(self, league_id):
        """Get league name from ID."""
        if self._league_cache is None:
            self._league_cache = self.get_league_mappings()
        
        if not self._league_cache.empty:
            league = self._league_cache[self._league_cache['id'] == league_id]
            if not league.empty:
                return league.iloc[0]['name']
        return f"League {league_id}"
    
    @st.cache_data(ttl=300)
    def fetch_data(_self, query, params=None):
        """Fetch data from database with caching."""
        try:
            with _self.engine.connect() as conn:
                if params:
                    # Convert date objects to strings for SQLAlchemy compatibility
                    processed_params = {}
                    for key, value in params.items():
                        if hasattr(value, 'strftime'):  # Handle date/datetime objects
                            processed_params[key] = value.strftime('%Y-%m-%d')
                        else:
                            processed_params[key] = value
                    result = pd.read_sql_query(text(query), conn, params=processed_params)
                else:
                    result = pd.read_sql_query(text(query), conn)
                return result
        except SQLAlchemyError as e:
            logger.error(f"Database query failed: {str(e)}")
            st.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()

class Dashboard:
    """Main dashboard class for football data visualization."""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state variables."""
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        if 'selected_league' not in st.session_state:
            st.session_state.selected_league = "All"
        if 'date_range' not in st.session_state:
            st.session_state.date_range = (
                datetime.now().date() - timedelta(days=30),
                datetime.now().date()
            )
    
    def run(self):
        """Main dashboard runner."""
        # Header
        st.markdown('<div class="main-header">⚽ Football Data Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Interactive visualization of football statistics and predictions</div>', unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        self.render_tabs()
    
    def render_sidebar(self):
        """Render sidebar with filters and controls."""
        st.sidebar.header("🎛️ Controls & Filters")
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.last_refresh = datetime.now()
            st.rerun()
        
        st.sidebar.markdown(f"*Last updated: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}*")
        
        # League filter
        leagues_df = self.db.fetch_data("SELECT name, country, season FROM leagues ORDER BY name")
        if not leagues_df.empty:
            league_options = ["All"] + [f"{row['name']} ({row['country']} {row['season']})" for _, row in leagues_df.iterrows()]
            selected_option = st.sidebar.selectbox(
                "🏆 Select League",
                league_options,
                index=0
            )
            # Extract just the league name for filtering
            st.session_state.selected_league = "All" if selected_option == "All" else selected_option.split(" (")[0]
        
        # Date range filter
        st.session_state.date_range = st.sidebar.date_input(
            "📅 Date Range",
            value=st.session_state.date_range,
            max_value=datetime.now().date()
        )
        
        # Summary statistics
        st.sidebar.header("📊 Quick Stats")
        self.render_sidebar_stats()
    
    def render_sidebar_stats(self):
        """Render sidebar statistics."""
        st.sidebar.header("📊 Quick Stats")
        
        # Recent fixtures
        st.sidebar.subheader("Recent Fixtures")
        fixtures_query = """
            SELECT f.home_team_id, f.away_team_id, f.match_date, l.name as league_name,
                   ht.name as home_team_name, at.name as away_team_name
            FROM fixtures f
            JOIN leagues l ON f.league_id = l.id
            JOIN teams ht ON f.home_team_id = ht.id
            JOIN teams at ON f.away_team_id = at.id
            WHERE f.match_date BETWEEN CURRENT_DATE - INTERVAL '7 days' AND CURRENT_DATE
            ORDER BY f.match_date DESC
            LIMIT 5
        """
        
        fixtures_df = self.db.fetch_data(fixtures_query)
        if not fixtures_df.empty:
            for _, fixture in fixtures_df.iterrows():
                st.sidebar.text(f"{fixture['home_team_name']} vs {fixture['away_team_name']}")
                st.sidebar.caption(f"{fixture['match_date'].strftime('%d %b')} | {fixture['league_name']}")
        else:
            st.sidebar.text("No recent fixtures")
        
        st.sidebar.divider()
        
        # Top teams
        st.sidebar.subheader("Top Teams")
        teams_query = """
            SELECT s.team_id, s.points, s.position, t.name as team_name, l.name as league_name
            FROM standings s
            JOIN teams t ON s.team_id = t.id
            JOIN leagues l ON s.league_id = l.id
            WHERE s.position <= 3
            ORDER BY s.league_id, s.position ASC
        """
        
        teams_df = self.db.fetch_data(teams_query)
        if not teams_df.empty:
            for _, team in teams_df.iterrows():
                st.sidebar.text(f"#{team['position']} - {team['team_name']}")
                st.sidebar.caption(f"{team['league_name']} - {team['points']} pts")
        else:
            st.sidebar.text("No standings data")
    
    def render_tabs(self):
        """Render main content tabs."""
        tabs = st.tabs([
            "🏆 Teams", 
            "📈 Standings", 
            "🔮 Predictions", 
            "🏅 Leagues", 
            "🔥 High Form Teams", 
            "📅 Fixtures", 
            "📋 Data Logs"
        ])
        
        with tabs[0]:
            self.render_teams_tab()
        with tabs[1]:
            self.render_standings_tab()
        with tabs[2]:
            self.render_predictions_tab()
        with tabs[3]:
            self.render_leagues_tab()
        with tabs[4]:
            self.render_high_form_tab()
        with tabs[5]:
            self.render_fixtures_tab()
        with tabs[6]:
            self.render_data_logs_tab()
    
    def render_teams_tab(self):
        """Render teams tab with data and visualizations."""
        st.header("🏆 Teams Overview")
        
        # Filter by league if selected
        query = "SELECT * FROM teams"
        params = None
        if st.session_state.selected_league != "All":
            query += " WHERE league = :league"
            params = {"league": st.session_state.selected_league}
        
        teams_df = self.db.fetch_data(query, params)
        
        if not teams_df.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Teams Data")
                st.dataframe(teams_df, use_container_width=True)
            
            with col2:
                st.subheader("Team Statistics")
                if 'points' in teams_df.columns:
                    fig = px.bar(
                        teams_df.head(10), 
                        x='name', 
                        y='points',
                        title="Top 10 Teams by Points"
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No teams data available.")
    
    def render_standings_tab(self):
        """Render standings tab with data and visualizations."""
        st.header("📈 League Standings")
        
        query = """
            SELECT s.*, l.name as league_name, t.name as team_name
            FROM standings s 
            JOIN leagues l ON s.league_id = l.id
            JOIN teams t ON s.team_id = t.id
        """
        params = None
        
        if st.session_state.selected_league != "All":
            query += " WHERE l.name = :league"
            params = {"league": st.session_state.selected_league}
        
        standings_df = self.db.fetch_data(query, params)
        
        if not standings_df.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Current Standings")
                display_df = standings_df[['position', 'team_name', 'played', 'won', 'drawn', 'lost', 'goals_for', 'goals_against', 'points']]
                display_df.columns = ['Pos', 'Team', 'P', 'W', 'D', 'L', 'GF', 'GA', 'Pts']
                st.dataframe(display_df, use_container_width=True)
            
            with col2:
                st.subheader("Points Distribution")
                if 'points' in standings_df.columns:
                    fig = px.histogram(
                        standings_df, 
                        x='points', 
                        nbins=10,
                        title="Points Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No standings data available.")
    
    def render_predictions_tab(self):
        """Render predictions tab with data and visualizations."""
        st.header("🔮 Match Predictions")
        
        query = "SELECT * FROM predictions"
        predictions_df = self.db.fetch_data(query)
        
        if not predictions_df.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Predictions Data")
                st.dataframe(predictions_df, use_container_width=True)
            
            with col2:
                st.subheader("Prediction Confidence")
                if 'confidence' in predictions_df.columns:
                    fig = px.scatter(
                        predictions_df, 
                        x='predicted_probability', 
                        y='confidence',
                        color='predicted_outcome',
                        title="Prediction Confidence vs Probability"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predictions data available.")
    
    def render_leagues_tab(self):
        """Render leagues tab with data and visualizations."""
        st.header("🏅 Leagues Overview")
        
        leagues_df = self.db.fetch_data("SELECT * FROM leagues")
        
        if not leagues_df.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Leagues Data")
                st.dataframe(leagues_df, use_container_width=True)
            
            with col2:
                st.subheader("League Distribution")
                fig = px.pie(
                    leagues_df, 
                    names='name', 
                    values='team_count' if 'team_count' in leagues_df.columns else None,
                    title="Teams per League"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No leagues data available.")
    
    def render_high_form_tab(self):
        """Render high form teams tab with data and visualizations."""
        st.header("🔥 High Form Teams")
        
        query = """
            SELECT h.*, t.name as team_name, l.name as league_name
            FROM high_form_teams h
            JOIN teams t ON h.team_id = t.id
            JOIN leagues l ON h.league_id = l.id
            ORDER BY h.win_rate DESC
            LIMIT 20
        """
        high_form_df = self.db.fetch_data(query)
        
        if not high_form_df.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("High Form Teams")
                # Display form teams with team names prominently
                display_df = high_form_df[['team_name', 'league_name', 'win_rate', 'wins', 'total_matches']]
                display_df.columns = ['Team', 'League', 'Win Rate', 'Wins', 'Total Matches']
                # Convert win_rate to percentage for display
                display_df['Win Rate'] = (display_df['Win Rate'] * 100).round(1).astype(str) + '%'
                st.dataframe(display_df, use_container_width=True)
            
            with col2:
                st.subheader("Form Metrics")
                if 'form_score' in high_form_df.columns:
                    fig = px.bar(
                        high_form_df.head(10),
                        x='team_name',
                        y='form_score',
                        title="Top 10 Form Scores"
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No high form teams data available.")
    
    def render_fixtures_tab(self):
        """Render fixtures tab with data and visualizations."""
        st.header("📅 Upcoming Fixtures")
        
        query = """
            SELECT 
                ht.name as home_team_name, 
                at.name as away_team_name,
                f.match_date,
                l.name as league_name
            FROM fixtures f 
            JOIN leagues l ON f.league_id = l.id
            JOIN teams ht ON f.home_team_id = ht.id
            JOIN teams at ON f.away_team_id = at.id
            WHERE f.match_date BETWEEN :start_date AND :end_date
            ORDER BY f.match_date ASC
        """
        fixtures_df = self.db.fetch_data(query, {
            "start_date": st.session_state.date_range[0], 
            "end_date": st.session_state.date_range[1]
        })
        
        if not fixtures_df.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Fixtures")
                # Create a clean display with only team names and relevant columns
                display_cols = ['home_team_name', 'away_team_name', 'match_date', 'league_name']
                
                # Ensure we have the required columns
                available_cols = [col for col in display_cols if col in fixtures_df.columns]
                if len(available_cols) >= 3:  # At least have team names and match_date
                    display_df = fixtures_df[available_cols].copy()
                    
                    # Rename columns for display
                    column_mapping = {
                        'home_team_name': 'Home Team',
                        'away_team_name': 'Away Team', 
                        'match_date': 'Match Date',
                        'league_name': 'League'
                    }
                    display_df = display_df.rename(columns=column_mapping)
                    
                    # Format the date
                    if 'Match Date' in display_df.columns:
                        display_df['Match Date'] = pd.to_datetime(display_df['Match Date']).dt.strftime('%Y-%m-%d %H:%M')
                    
                    st.dataframe(display_df, use_container_width=True)
                else:
                    # Fallback - show basic info without team IDs
                    basic_cols = ['match_date', 'league_name']
                    basic_available = [col for col in basic_cols if col in fixtures_df.columns]
                    if basic_available:
                        basic_display = fixtures_df[basic_available].copy()
                        st.dataframe(basic_display, use_container_width=True)
                    else:
                        st.dataframe(fixtures_df[['home_team_name', 'away_team_name', 'match_date', 'league_name']], use_container_width=True)
            
            with col2:
                st.subheader("Fixtures by Date")
                if 'match_date' in fixtures_df.columns:
                    fixtures_df['match_date'] = pd.to_datetime(fixtures_df['match_date'])
                    fixtures_by_date = fixtures_df.groupby(fixtures_df['match_date'].dt.date).size().reset_index(name='count')
                    fig = px.bar(
                        fixtures_by_date,
                        x='match_date',
                        y='count',
                        title="Fixtures by Date"
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No fixtures found for the selected date range.")
    
    def render_data_logs_tab(self):
        """Render data fetch logs tab with data and visualizations."""
        st.header("📋 Data Fetch Logs")
        
        query = """
            SELECT * FROM data_fetch_logs 
            WHERE fetch_date BETWEEN :start_date AND :end_date
            ORDER BY fetch_date DESC
        """
        logs_df = self.db.fetch_data(query, {
            "start_date": st.session_state.date_range[0], 
            "end_date": st.session_state.date_range[1]
        })
        
        if not logs_df.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Fetch Logs")
                st.dataframe(logs_df, use_container_width=True)
            
            with col2:
                st.subheader("Fetch Timeline")
                if 'fetch_date' in logs_df.columns:
                    logs_df['fetch_date'] = pd.to_datetime(logs_df['fetch_date'])
                    success_counts = logs_df['status'].value_counts()
                    fig = px.pie(
                        values=success_counts.values,
                        names=success_counts.index,
                        title="Fetch Success Rate"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data fetch logs found for the selected date range.")

def main():
    """Main application entry point."""
    try:
        dashboard = Dashboard()
        dashboard.run()
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        st.error("An error occurred while loading the dashboard. Please check the logs.")

if __name__ == "__main__":
    main()