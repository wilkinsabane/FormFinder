"""
Football Data Dashboard — improved single-file Streamlit app.

Drop-in replacement for the older script. Key features:
- Robust DB connection (SQLAlchemy engine cached)
- Cached queries with st.cache_data
- Retry on transient DB failures
- Safe parameterized queries
- Pagination + CSV download
- Cleaner UI controls and better filter handling
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import os
import logging
from datetime import datetime, timedelta, date

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError

# ---------- Configuration & Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("football_dashboard")

@dataclass
class Config:
    db_uri: Optional[str] = None
    default_days: int = 30

    @classmethod
    def load(cls) -> "Config":
        # Prefer Streamlit secrets if available
        secrets_uri = None
        try:
            secrets_uri = st.secrets.get("DB_URI") if hasattr(st, "secrets") else None
        except Exception:
            secrets_uri = None
        env_uri = os.getenv("DB_URI")
        db_uri = secrets_uri or env_uri
        return cls(db_uri=db_uri)


# ---------- Utility helpers ----------
def retry_on_exception(fn, retries=3, delay_seconds=1):
    """Simple retry wrapper for transient DB errors."""
    import time
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except OperationalError as e:
            logger.warning(f"OperationalError (attempt {attempt}/{retries}): {e}")
            if attempt == retries:
                raise
            time.sleep(delay_seconds)
        except Exception:
            # Let other errors bubble up
            raise


def fmt_dt(val):
    if pd.isna(val):
        return ""
    if isinstance(val, (datetime, date)):
        return val.strftime("%Y-%m-%d %H:%M") if isinstance(val, datetime) else val.strftime("%Y-%m-%d")
    try:
        return pd.to_datetime(val).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(val)


# ---------- Database manager ----------
class DatabaseManager:
    """Manages SQLAlchemy engine and query execution."""

    def __init__(self, db_uri: str):
        if not db_uri:
            raise ValueError("DB_URI not provided. Set DB_URI in Streamlit secrets or environment variables.")
        self.db_uri = db_uri
        self.engine = self._get_engine(self.db_uri)

    @st.cache_resource(show_spinner=False)
    def _get_engine(_self, db_uri: str):
        # create_engine is cached by Streamlit so it's created once per session
        engine = create_engine(
            db_uri,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            future=True,
        )
        # simple smoke test
        def _test():
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        retry_on_exception(_test, retries=2, delay_seconds=0.5)
        logger.info("DB engine created and tested")
        return engine

    def fetch_df(self, query: str, params: Optional[Dict[str, Any]] = None, parse_dates: Optional[list] = None) -> pd.DataFrame:
        """Fetch dataframe from DB with safe params. Returns empty df on error (and logs)."""
        try:
            def _run():
                with self.engine.connect() as conn:
                    # pandas read_sql uses param style appropriate to SQLAlchemy
                    df = pd.read_sql_query(text(query), conn, params=params)
                    if parse_dates and not df.empty:
                        for col in parse_dates:
                            if col in df.columns:
                                df[col] = pd.to_datetime(df[col], errors="coerce")
                    return df
            return retry_on_exception(_run, retries=2, delay_seconds=0.5)
        except Exception as e:
            logger.exception("Error fetching data from DB")
            st.error(f"Database error: {str(e)}")
            return pd.DataFrame()


# ---------- Cached helpers for mappings ----------
@st.cache_data(ttl=600, show_spinner=False)
def load_team_mappings(_db: DatabaseManager) -> pd.DataFrame:
    q = "SELECT id, name, short_name FROM teams ORDER BY name"
    return _db.fetch_df(q)


@st.cache_data(ttl=600, show_spinner=False)
def load_league_mappings(_db: DatabaseManager) -> pd.DataFrame:
    q = "SELECT id, name, country, season FROM leagues ORDER BY name"
    return _db.fetch_df(q)


# ---------- Dashboard class ----------
class Dashboard:
    def __init__(self, db: DatabaseManager, cfg: Config):
        self.db = db
        self.cfg = cfg
        self.init_session_state()
        self.teams_cache = load_team_mappings(self.db)
        self.leagues_cache = load_league_mappings(self.db)

    def init_session_state(self):
        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        if "selected_league_ids" not in st.session_state:
            st.session_state.selected_league_ids = []
        if "selected_team_ids" not in st.session_state:
            st.session_state.selected_team_ids = []
        if "date_range" not in st.session_state:
            end = datetime.now().date()
            start = end - timedelta(days=self.cfg.default_days)
            st.session_state.date_range = (start, end)

    def run(self):
        st.set_page_config(page_title="Football Data Dashboard", page_icon="⚽", layout="wide")
        st.header("⚽ Football Data Dashboard")
        st.caption("Interactive visualizations from PostgreSQL — improved and more robust")

        # Controls
        with st.sidebar:
            st.subheader("Controls & Filters")
            if st.button("🔄 Refresh data (clear caches)"):
                # clear caches by calling functions with .clear? Use Streamlit cache clearing
                try:
                    load_team_mappings.clear()
                    load_league_mappings.clear()
                    st.cache_data.clear()
                    st.session_state.last_refresh = datetime.now()
                    st.experimental_rerun()
                except Exception:
                    st.warning("Could not clear caches programmatically in this environment. Reload the app to ensure fresh data.")
            st.markdown(f"**Last refresh:** {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")

            # Date range
            start, end = st.date_input("📅 Date range", value=st.session_state.date_range, max_value=datetime.now().date())
            st.session_state.date_range = (start, end)

            # League multi-select (use IDs for safety)
            leagues_df = self.leagues_cache
            league_options = []
            if not leagues_df.empty:
                league_options = list(leagues_df.apply(lambda r: (r["id"], f"{r['name']} ({r['country']} {r['season']})"), axis=1))
                ids, labels = zip(*league_options)
                selected = st.multiselect("🏆 League(s)", options=ids, format_func=lambda i, lbl_map=dict(league_options): dict(league_options).get(i, str(i)))
                st.session_state.selected_league_ids = selected

            # Team multi-select
            teams_df = self.teams_cache
            if not teams_df.empty:
                team_options = list(teams_df.apply(lambda r: (r["id"], r["name"]), axis=1))
                ids, labels = zip(*team_options)
                selected_teams = st.multiselect("👥 Team(s)", options=ids, format_func=lambda i, tm_map=dict(team_options): dict(team_options).get(i, str(i)))
                st.session_state.selected_team_ids = selected_teams

        # Tabs
        tabs = st.tabs(["Teams", "Standings", "Fixtures", "Leagues", "Predictions", "High Form", "Data Logs"])

        with tabs[0]:
            self.teams_tab()

        with tabs[1]:
            self.standings_tab()

        with tabs[2]:
            self.fixtures_tab()

        with tabs[3]:
            self.leagues_tab()

        with tabs[4]:
            self.predictions_tab()

        with tabs[5]:
            self.high_form_tab()

        with tabs[6]:
            self.data_logs_tab()

    # ---------- Tab implementations ----------
    def teams_tab(self):
        st.subheader("Teams")
        params = {}
        query = "SELECT * FROM teams"
        filters = []
        if st.session_state.selected_team_ids:
            filters.append("id = ANY(:team_ids)")
            params["team_ids"] = list(st.session_state.selected_team_ids)
        if st.session_state.selected_league_ids:
            filters.append("league_id = ANY(:league_ids)")
            params["league_ids"] = list(st.session_state.selected_league_ids)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        df = self.db.fetch_df(query, params=params)
        self._show_dataframe_with_tools(df, page_size=25, filename="teams.csv")

        if not df.empty and "points" in df.columns:
            top = df.sort_values(by="points", ascending=False).head(10)
            fig = px.bar(top, x="name", y="points", title="Top 10 Teams by Points")
            st.plotly_chart(fig, use_container_width=True)

    def standings_tab(self):
        st.subheader("Standings")
        query = """
            SELECT s.*, l.name as league_name, t.name as team_name
            FROM standings s
            JOIN leagues l ON s.league_id = l.id
            JOIN teams t ON s.team_id = t.id
        """
        params = {}
        if st.session_state.selected_league_ids:
            query += " WHERE s.league_id = ANY(:league_ids)"
            params["league_ids"] = list(st.session_state.selected_league_ids)
        df = self.db.fetch_df(query, params=params)
        if df.empty:
            st.info("No standings available.")
            return
        display_cols = ["position", "team_name", "played", "won", "drawn", "lost", "goals_for", "goals_against", "points"]
        available = [c for c in display_cols if c in df.columns]
        df_display = df[available].copy()
        df_display.columns = ["Pos" if c=="position" else c.title().replace("_"," ") for c in available]
        self._show_dataframe_with_tools(df_display, page_size=25, filename="standings.csv")

        if "points" in df.columns:
            fig = px.histogram(df, x="points", nbins=10, title="Points Distribution")
            st.plotly_chart(fig, use_container_width=True)

    def fixtures_tab(self):
        st.subheader("Fixtures")
        q = """
            SELECT ht.name as home_team, at.name as away_team, f.match_date, l.name as league_name
            FROM fixtures f
            JOIN teams ht ON f.home_team_id = ht.id
            JOIN teams at ON f.away_team_id = at.id
            JOIN leagues l ON f.league_id = l.id
            WHERE f.match_date BETWEEN :start_date AND :end_date
            ORDER BY f.match_date ASC
        """
        start_date = st.session_state.date_range[0]
        end_date = st.session_state.date_range[1]
        df = self.db.fetch_df(q, params={"start_date": start_date, "end_date": end_date}, parse_dates=["match_date"])
        if df.empty:
            st.info("No fixtures in the selected date range.")
            return
        # format date column for readability
        if "match_date" in df.columns:
            df["match_date"] = df["match_date"].apply(fmt_dt)
        self._show_dataframe_with_tools(df, page_size=20, filename="fixtures.csv")

        # fixtures by date chart
        if "match_date" in df.columns:
            df_chart = df.copy()
            df_chart["match_date"] = pd.to_datetime(df_chart["match_date"], errors="coerce").dt.date
            summary = df_chart.groupby("match_date").size().reset_index(name="count")
            fig = px.bar(summary, x="match_date", y="count", title="Fixtures by Date")
            st.plotly_chart(fig, use_container_width=True)

    def leagues_tab(self):
        st.subheader("Leagues")
        q = "SELECT * FROM leagues"
        df = self.db.fetch_df(q)
        self._show_dataframe_with_tools(df, page_size=25, filename="leagues.csv")
        if not df.empty:
            # if team_count exists use it, else show distribution by country
            if "team_count" in df.columns:
                fig = px.pie(df, names="name", values="team_count", title="Teams per League")
            else:
                fig = px.histogram(df, x="country", title="Leagues by Country")
            st.plotly_chart(fig, use_container_width=True)

    def predictions_tab(self):
        st.subheader("Predictions")
        q = "SELECT * FROM predictions ORDER BY created_at DESC"
        df = self.db.fetch_df(q)
        self._show_dataframe_with_tools(df, page_size=25, filename="predictions.csv")
        if not df.empty and {"predicted_probability","confidence","predicted_outcome"}.issubset(df.columns):
            fig = px.scatter(df, x="predicted_probability", y="confidence", color="predicted_outcome", title="Prediction Confidence vs Probability")
            st.plotly_chart(fig, use_container_width=True)

    def high_form_tab(self):
        st.subheader("High Form Teams")
        q = """
            SELECT h.*, t.name as team_name, l.name as league_name
            FROM high_form_teams h
            JOIN teams t ON h.team_id = t.id
            JOIN leagues l ON h.league_id = l.id
            ORDER BY h.win_rate DESC
            LIMIT 200
        """
        df = self.db.fetch_df(q)
        if df.empty:
            st.info("No high-form data.")
            return
        # display with percentage conversion if win_rate present
        if "win_rate" in df.columns:
            df = df.copy()
            try:
                df["win_rate_pct"] = (df["win_rate"].astype(float) * 100).round(1).astype(str) + "%"
            except Exception:
                pass
        self._show_dataframe_with_tools(df, page_size=25, filename="high_form.csv")
        if "form_score" in df.columns:
            fig = px.bar(df.head(15), x="team_name", y="form_score", title="Top Form Scores")
            st.plotly_chart(fig, use_container_width=True)

    def data_logs_tab(self):
        st.subheader("Data Fetch Logs")
        q = """
            SELECT * FROM data_fetch_logs
            WHERE fetch_date BETWEEN :start_date AND :end_date
            ORDER BY fetch_date DESC
        """
        start_date, end_date = st.session_state.date_range
        df = self.db.fetch_df(q, params={"start_date": start_date, "end_date": end_date}, parse_dates=["fetch_date"])
        if df.empty:
            st.info("No fetch logs for the selected range.")
            return
        if "fetch_date" in df.columns:
            df["fetch_date"] = df["fetch_date"].apply(fmt_dt)
        self._show_dataframe_with_tools(df, page_size=30, filename="fetch_logs.csv")
        if "status" in df.columns:
            counts = df["status"].value_counts()
            fig = px.pie(names=counts.index, values=counts.values, title="Fetch Status Distribution")
            st.plotly_chart(fig, use_container_width=True)

    # ---------- small helper to show dataframes + download/pagination ----------
    def _show_dataframe_with_tools(self, df: pd.DataFrame, page_size: int = 20, filename: str = "data.csv"):
        if df is None or df.empty:
            st.write("No data to show.")
            return
        total = len(df)
        st.write(f"Showing {total:,} rows")
        # pagination controls
        page = st.number_input(
            "Page",
            min_value=1,
            value=1,
            step=1,
            max_value=max(1, (total - 1) // page_size + 1),
            key=f"page_picker_{filename}",
        )
        start_idx = (page - 1) * page_size
        end_idx = min(total, start_idx + page_size)
        sub = df.iloc[start_idx:end_idx]
        st.dataframe(sub, use_container_width=True)
        # download
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download full CSV",
            data=csv,
            file_name=filename,
            mime="text/csv",
            key=f"download_button_{filename}",
        )
        # quick copy first rows
        st.code(df.head(5).to_json(orient="records", date_format="iso"), language="json")


# ---------- Entrypoint ----------
def main():
    try:
        cfg = Config.load()
        if not cfg.db_uri:
            st.error("Database URI missing. Set DB_URI in Streamlit secrets or environment variables.")
            st.stop()

        db = DatabaseManager(cfg.db_uri)
        dashboard = Dashboard(db, cfg)
        dashboard.run()

    except Exception as e:
        logger.exception("Unhandled error in dashboard")
        st.error("An unexpected error occurred. Check logs for details.")


if __name__ == "__main__":
    main()
