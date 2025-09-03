Awesome — here’s a clean, production-ready **scaffolding** you can drop into your repo. It matches the PRD v3.0 you approved, uses your exact H2H/Preview field names, and keeps things modular, cache-aware, and testable.

I’ve organized this as file-by-file snippets. You can paste them directly (keep the paths).

---

# 1) DB Migration — `migrations/20250815_h2h_cache_and_predictions.sql`

```sql
-- 1) H2H cache table (24h TTL policy handled in code)
CREATE TABLE IF NOT EXISTS h2h_cache (
  team1_id INTEGER NOT NULL,
  team2_id INTEGER NOT NULL,
  competition_id INTEGER,
  -- overall
  overall_games_played INTEGER NOT NULL,
  overall_team1_wins INTEGER NOT NULL,
  overall_team2_wins INTEGER NOT NULL,
  overall_draws INTEGER NOT NULL,
  overall_team1_scored INTEGER NOT NULL,
  overall_team2_scored INTEGER NOT NULL,
  -- team1 at home
  team1_games_played_at_home INTEGER,
  team1_wins_at_home INTEGER,
  team1_losses_at_home INTEGER,
  team1_draws_at_home INTEGER,
  team1_scored_at_home INTEGER,
  team1_conceded_at_home INTEGER,
  -- team2 at home
  team2_games_played_at_home INTEGER,
  team2_wins_at_home INTEGER,
  team2_losses_at_home INTEGER,
  team2_draws_at_home INTEGER,
  team2_scored_at_home INTEGER,
  team2_conceded_at_home INTEGER,
  -- computed convenience
  avg_total_goals FLOAT NOT NULL,
  -- caching metadata
  etag TEXT,
  last_fetched_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (team1_id, team2_id, COALESCE(competition_id, 0))
);
CREATE INDEX IF NOT EXISTS h2h_cache_last_fetched_idx ON h2h_cache (last_fetched_at DESC);

-- 2) Predictions table alterations for goal-based outputs
ALTER TABLE predictions
  DROP COLUMN IF EXISTS home_win_probability,
  DROP COLUMN IF EXISTS draw_probability,
  DROP COLUMN IF EXISTS away_win_probability,
  ADD COLUMN IF NOT EXISTS predicted_total_goals FLOAT,
  ADD COLUMN IF NOT EXISTS over_2_5_probability FLOAT,
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;
```

---

# 2) Config & Logging

## 2.1 `formfinder/config.py` (minimal additions)

```python
from pydantic import BaseModel
from functools import lru_cache
import os

class APIConfig(BaseModel):
    base_url: str = "https://api.soccerdataapi.com"
    h2h_path: str = "/head-to-head/"
    preview_path: str = "/match-preview/"
    auth_token: str = os.getenv("SOCCERDATA_API_KEY", "")
    timeout: int = 15
    rate_limit_requests: int = 300  # requests / period
    rate_limit_period: int = 60     # seconds
    max_retries: int = 3
    h2h_ttl_seconds: int = 24 * 3600
    preview_ttl_seconds: int = 2 * 3600

class LoggingConfig(BaseModel):
    level: str = "INFO"
    json: bool = False
    log_to_file: bool = True
    file_path: str = "logs/formfinder.log"

class DBConfig(BaseModel):
    url: str = os.getenv("DATABASE_URL", "postgresql+psycopg2://user:pass@localhost:5432/formfinder")

class FormFinderConfig(BaseModel):
    api: APIConfig = APIConfig()
    logging: LoggingConfig = LoggingConfig()
    db: DBConfig = DBConfig()

@lru_cache
def get_config() -> FormFinderConfig:
    return FormFinderConfig()
```

## 2.2 `formfinder/logger.py`

```python
import logging, os, json
from .config import get_config

def setup_logging() -> logging.Logger:
    cfg = get_config().logging
    logger = logging.getLogger("FormFinder")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))

    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
    if cfg.json:
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                base = {
                    "ts": self.formatTime(record, self.datefmt),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                for k, v in getattr(record, "__dict__", {}).items():
                    if k.startswith(("fixture_", "team_", "cache_", "api_", "req_")):
                        base[k] = v
                return json.dumps(base)
        fmt = JsonFormatter()

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if cfg.log_to_file:
        os.makedirs(os.path.dirname(cfg.file_path), exist_ok=True)
        fh = logging.FileHandler(cfg.file_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

log = setup_logging()
```

## 2.3 `formfinder/exceptions.py`

```python
class APIError(Exception):
    """Raised when an API call fails after retries or receives an invalid response."""
    pass

class FeatureError(Exception):
    """Raised when feature extraction/parsing fails or fields are missing."""
    pass
```

---

# 3) API Client (rate-limit, retries, ETag, DB cache)

## `formfinder/clients/api_client.py`

```python
from __future__ import annotations
import time, requests, math
from typing import Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..config import get_config
from ..logger import log
from ..exceptions import APIError

class SoccerDataAPIClient:
    def __init__(self, db_session: Session):
        self.cfg = get_config().api
        self.session = requests.Session()
        self.session.headers.update({
            "Accept-Encoding": "gzip",
            "Content-Type": "application/json",
        })
        self.db = db_session

    def _rate_limit_sleep(self, attempt: int, retry_after: Optional[float] = None):
        if retry_after:
            time.sleep(retry_after)
        else:
            # exponential backoff starting ~1s
            time.sleep(min(2 ** attempt, 16))

    def _base_url(self, path: str) -> str:
        return f"{self.cfg.base_url.rstrip('/')}{path}"

    def _get(self, url: str, params: dict, etag: Optional[str] = None) -> requests.Response:
        p = dict(params or {})
        # The API examples pass auth_token as a query parameter
        p["auth_token"] = self.cfg.auth_token
        headers = {}
        if etag:
            headers["If-None-Match"] = etag
        return self.session.get(url, params=p, headers=headers, timeout=self.cfg.timeout)

    # ---------- H2H ----------
    def get_h2h_stats(self, team1_id: int, team2_id: int, competition_id: Optional[int] = None) -> Dict[str, Any]:
        """Return flattened H2H dict (includes advanced splits).
        Cache key: (team1_id, team2_id, competition_id|0), TTL 24h.
        """
        # 1) cache lookup
        key_comp = competition_id or 0
        row = self.db.execute(
            text("""
                SELECT *, EXTRACT(EPOCH FROM (NOW() - last_fetched_at)) AS age
                FROM h2h_cache
                WHERE team1_id=:t1 AND team2_id=:t2 AND COALESCE(competition_id,0)=:c
            """),
            {"t1": team1_id, "t2": team2_id, "c": key_comp}
        ).mappings().first()

        if row and row["age"] is not None and row["age"] < self.cfg.h2h_ttl_seconds:
            log.info("H2H CACHE HIT", extra={"team1": team1_id, "team2": team2_id, "cache_hit": True})
            return dict(row)

        log.info("H2H CACHE MISS", extra={"team1": team1_id, "team2": team2_id, "cache_hit": False})

        # 2) conditional GET using stored etag if present
        etag = row["etag"] if row and row.get("etag") else None
        url = self._base_url(self.cfg.h2h_path)
        params = {"team_1_id": team1_id, "team_2_id": team2_id}
        if competition_id:
            params["competition_id"] = competition_id

        response = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                response = self._get(url, params, etag=etag)
                if response.status_code == 304 and row:
                    # Not modified → refresh timestamp only
                    self.db.execute(
                        text("""UPDATE h2h_cache SET last_fetched_at=NOW() 
                                WHERE team1_id=:t1 AND team2_id=:t2 AND COALESCE(competition_id,0)=:c"""),
                        {"t1": team1_id, "t2": team2_id, "c": key_comp}
                    )
                    self.db.commit()
                    return dict(row)

                if response.status_code == 429:
                    ra = float(response.headers.get("Retry-After", "1"))
                    self._rate_limit_sleep(attempt, retry_after=ra)
                    continue

                if response.status_code >= 500:
                    self._rate_limit_sleep(attempt)
                    continue

                if response.status_code != 200:
                    raise APIError(f"H2H HTTP {response.status_code}: {response.text}")

                data = response.json()
                break
            except requests.RequestException as e:
                if attempt >= self.cfg.max_retries:
                    raise APIError(f"H2H request failed: {e}")
                self._rate_limit_sleep(attempt)

        # 3) flatten fields (exact names from your example)
        stats = data.get("stats", {})
        overall = stats.get("overall", {}) or {}
        t1h    = stats.get("team1_at_home", {}) or {}
        t2h    = stats.get("team2_at_home", {}) or {}

        overall_games_played   = int(overall.get("overall_games_played", 0))
        overall_team1_scored   = int(overall.get("overall_team1_scored", 0))
        overall_team2_scored   = int(overall.get("overall_team2_scored", 0))

        avg_total_goals = (overall_team1_scored + overall_team2_scored) / overall_games_played if overall_games_played else 0.0

        flat = {
            "team1_id": team1_id,
            "team2_id": team2_id,
            "competition_id": competition_id,
            "overall_games_played": overall_games_played,
            "overall_team1_wins": int(overall.get("overall_team1_wins", 0)),
            "overall_team2_wins": int(overall.get("overall_team2_wins", 0)),
            "overall_draws": int(overall.get("overall_draws", 0)),
            "overall_team1_scored": overall_team1_scored,
            "overall_team2_scored": overall_team2_scored,
            # team1 at home
            "team1_games_played_at_home": t1h.get("team1_games_played_at_home"),
            "team1_wins_at_home": t1h.get("team1_wins_at_home"),
            "team1_losses_at_home": t1h.get("team1_losses_at_home"),
            "team1_draws_at_home": t1h.get("team1_draws_at_home"),
            "team1_scored_at_home": t1h.get("team1_scored_at_home"),
            "team1_conceded_at_home": t1h.get("team1_conceded_at_home"),
            # team2 at home
            "team2_games_played_at_home": t2h.get("team2_games_played_at_home"),
            "team2_wins_at_home": t2h.get("team2_wins_at_home"),
            "team2_losses_at_home": t2h.get("team2_losses_at_home"),
            "team2_draws_at_home": t2h.get("team2_draws_at_home"),
            "team2_scored_at_home": t2h.get("team2_scored_at_home"),
            "team2_conceded_at_home": t2h.get("team2_conceded_at_home"),
            # computed
            "avg_total_goals": float(avg_total_goals),
        }

        # 4) UPSERT to cache
        etag_new = response.headers.get("ETag")
        flat_with_meta = dict(flat)
        flat_with_meta["etag"] = etag_new
        self.db.execute(text("""
            INSERT INTO h2h_cache (
              team1_id, team2_id, competition_id,
              overall_games_played, overall_team1_wins, overall_team2_wins, overall_draws,
              overall_team1_scored, overall_team2_scored,
              team1_games_played_at_home, team1_wins_at_home, team1_losses_at_home, team1_draws_at_home,
              team1_scored_at_home, team1_conceded_at_home,
              team2_games_played_at_home, team2_wins_at_home, team2_losses_at_home, team2_draws_at_home,
              team2_scored_at_home, team2_conceded_at_home,
              avg_total_goals, etag, last_fetched_at
            ) VALUES (
              :team1_id, :team2_id, :competition_id,
              :overall_games_played, :overall_team1_wins, :overall_team2_wins, :overall_draws,
              :overall_team1_scored, :overall_team2_scored,
              :team1_games_played_at_home, :team1_wins_at_home, :team1_losses_at_home, :team1_draws_at_home,
              :team1_scored_at_home, :team1_conceded_at_home,
              :team2_games_played_at_home, :team2_wins_at_home, :team2_losses_at_home, :team2_draws_at_home,
              :team2_scored_at_home, :team2_conceded_at_home,
              :avg_total_goals, :etag, NOW()
            )
            ON CONFLICT (team1_id, team2_id, COALESCE(competition_id, 0))
            DO UPDATE SET
              overall_games_played=EXCLUDED.overall_games_played,
              overall_team1_wins=EXCLUDED.overall_team1_wins,
              overall_team2_wins=EXCLUDED.overall_team2_wins,
              overall_draws=EXCLUDED.overall_draws,
              overall_team1_scored=EXCLUDED.overall_team1_scored,
              overall_team2_scored=EXCLUDED.overall_team2_scored,
              team1_games_played_at_home=EXCLUDED.team1_games_played_at_home,
              team1_wins_at_home=EXCLUDED.team1_wins_at_home,
              team1_losses_at_home=EXCLUDED.team1_losses_at_home,
              team1_draws_at_home=EXCLUDED.team1_draws_at_home,
              team1_scored_at_home=EXCLUDED.team1_scored_at_home,
              team1_conceded_at_home=EXCLUDED.team1_conceded_at_home,
              team2_games_played_at_home=EXCLUDED.team2_games_played_at_home,
              team2_wins_at_home=EXCLUDED.team2_wins_at_home,
              team2_losses_at_home=EXCLUDED.team2_losses_at_home,
              team2_draws_at_home=EXCLUDED.team2_draws_at_home,
              team2_scored_at_home=EXCLUDED.team2_scored_at_home,
              team2_conceded_at_home=EXCLUDED.team2_conceded_at_home,
              avg_total_goals=EXCLUDED.avg_total_goals,
              etag=EXCLUDED.etag,
              last_fetched_at=NOW();
        """), flat_with_meta)
        self.db.commit()

        return flat

    # ---------- Match Preview ----------
    def get_match_preview(self, match_id: int) -> Dict[str, Any]:
        """Return preview dict with excitement_rating and content list. Short TTL cached is handled by caller if desired."""
        url = self._base_url(self.cfg.preview_path)
        params = {"match_id": match_id}

        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self._get(url, params)
                if resp.status_code == 429:
                    ra = float(resp.headers.get("Retry-After", "1"))
                    self._rate_limit_sleep(attempt, retry_after=ra)
                    continue
                if resp.status_code >= 500:
                    self._rate_limit_sleep(attempt)
                    continue
                if resp.status_code != 200:
                    raise APIError(f"Preview HTTP {resp.status_code}: {resp.text}")
                return resp.json()
            except requests.RequestException as e:
                if attempt >= self.cfg.max_retries:
                    raise APIError(f"Preview request failed: {e}")
                self._rate_limit_sleep(attempt)
```

> Note: Preview short-TTL caching can be added to DB if you want; many teams just keep it in an in-memory cache layer for a few hours.

---

# 4) Feature Engineering (pure functions)

## `formfinder/features.py`

```python
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
    row = db_session.execute(q, {"team_id": team_id, "match_date": match_date, "limit": last_n_games}).mappings().first()
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

    if compute_sentiment:
        try:
            content = " ".join([c.get("content", "") for c in (data.get("content") or [])])
            # plug-in your sentiment lib here; placeholder zero
            sentiment = 0.0
            result["preview_sentiment"] = float(sentiment)
        except Exception:
            result["preview_sentiment"] = 0.0

    return result
```

---

# 5) Training Pipeline Skeleton

## `scripts/train_model.py`

```python
from __future__ import annotations
import os, glob, json
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, brier_score_loss
from xgboost import XGBRegressor, XGBClassifier
from joblib import dump
from datetime import datetime

from formfinder.config import get_config
from formfinder.logger import log
from formfinder.clients.api_client import SoccerDataAPIClient
from formfinder.features import get_rolling_form_features, get_h2h_feature, get_preview_metrics

def load_leagues(path="free_leagues.txt") -> list[int]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [int(x.strip()) for x in f if x.strip().isdigit()]

def main():
    cfg = get_config()
    engine = create_engine(cfg.db.url, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)
    db = Session()

    leagues = load_leagues()
    api = SoccerDataAPIClient(db_session=db)

    # --- assemble dataset ---
    rows = db.execute(text("""
        SELECT fixture_id, league_id, match_date, home_team_id, away_team_id, home_score, away_score
        FROM fixtures
        WHERE status='finished'
          AND (:use_leagues = 0 OR league_id = ANY(:league_ids))
        ORDER BY match_date ASC
    """), {"use_leagues": 0 if not leagues else 1, "league_ids": leagues}).mappings().all()

    feats = []
    for r in rows:
        try:
            home_form = get_rolling_form_features(r["home_team_id"], r["match_date"], db)
            away_form = get_rolling_form_features(r["away_team_id"], r["match_date"], db)
            h2h = get_h2h_feature(r["home_team_id"], r["away_team_id"], api_client=api)
            # preview: optional for historical (only if available by id)
            try:
                preview = get_preview_metrics(r["fixture_id"], api_client=api)
            except Exception:
                preview = {"preview_excitement_rating": 0.0}

            x = {
                "league_id": r["league_id"],
                **{f"home_{k}": v for k, v in home_form.items()},
                **{f"away_{k}": v for k, v in away_form.items()},
                **h2h,
                **preview,
            }
            y_total = float((r["home_score"] or 0) + (r["away_score"] or 0))
            y_over25 = 1.0 if y_total > 2.5 else 0.0
            feats.append((r["match_date"], x, y_total, y_over25))
        except Exception as e:
            log.warning(f"feature_build_failed fixture={r['fixture_id']} err={e}")

    if not feats:
        log.error("No training data built; aborting.")
        return

    # sort by date, then expand to DataFrame
    feats.sort(key=lambda t: t[0])
    X = pd.DataFrame([f[1] for f in feats]).fillna(0.0)
    y_total = pd.Series([f[2] for f in feats])
    y_over25 = pd.Series([f[3] for f in feats])

    # --- modeling ---
    tscv = TimeSeriesSplit(n_splits=5)
    reg = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42
    )
    clf = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric="logloss"
    )

    # (simple CV example; feel free to expand with proper evaluation storage)
    # Fit on all (after quick sanity check split)
    reg.fit(X, y_total)
    clf.fit(X, y_over25)

    # quick sanity metrics
    pred_total = reg.predict(X)
    mae = mean_absolute_error(y_total, pred_total)
    prob_over25 = clf.predict_proba(X)[:, 1]
    brier = brier_score_loss(y_over25, prob_over25)
    log.info(f"Training complete: MAE={mae:.3f} Brier={brier:.3f}")

    # save artifacts
    os.makedirs("models", exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M")
    reg_path = f"models/goal_regressor_{stamp}.joblib"
    clf_path = f"models/over25_classifier_{stamp}.joblib"
    dump(reg, reg_path)
    dump(clf, clf_path)
    meta = {
        "created_utc": stamp,
        "features": list(X.columns),
        "rows": len(X),
        "metrics": {"mae": float(mae), "brier": float(brier)},
        "note": "H2H field names reflect team1/team2 schema; avg_total_goals computed."
    }
    with open(f"models/metadata_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Saved: {reg_path}, {clf_path}")

if __name__ == "__main__":
    main()
```

---

# 6) Prediction Writer (refactor snippet)

## `standalone_predictor_outputter.py` (core logic section)

```python
import os, glob, json
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from joblib import load
from datetime import datetime

from formfinder.config import get_config
from formfinder.logger import log
from formfinder.clients.api_client import SoccerDataAPIClient
from formfinder.features import get_rolling_form_features, get_h2h_feature, get_preview_metrics

def _latest(path_glob: str) -> str:
    files = glob.glob(path_glob)
    return max(files, key=os.path.getmtime) if files else ""

def main():
    cfg = get_config()
    engine = create_engine(cfg.db.url, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)
    db = Session()
    api = SoccerDataAPIClient(db_session=db)

    reg_path = _latest("models/goal_regressor_*.joblib")
    clf_path = _latest("models/over25_classifier_*.joblib")
    meta_path = _latest("models/metadata_*.json")
    if not (reg_path and clf_path and meta_path):
        log.error("Model artifacts not found.")
        return

    reg = load(reg_path)
    clf = load(clf_path)
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    feature_order = meta["features"]

    # upcoming fixtures
    rows = db.execute(text("""
        SELECT fixture_id, league_id, match_date, home_team_id, away_team_id
        FROM fixtures
        WHERE status != 'finished'
        ORDER BY match_date ASC
    """)).mappings().all()

    for r in rows:
        try:
            home_form = get_rolling_form_features(r["home_team_id"], r["match_date"], db)
            away_form = get_rolling_form_features(r["away_team_id"], r["match_date"], db)
            h2h = get_h2h_feature(r["home_team_id"], r["away_team_id"], api_client=api)
            try:
                preview = get_preview_metrics(r["fixture_id"], api_client=api)
            except Exception:
                preview = {"preview_excitement_rating": 0.0}

            x = {
                "league_id": r["league_id"],
                **{f"home_{k}": v for k, v in home_form.items()},
                **{f"away_{k}": v for k, v in away_form.items()},
                **h2h,
                **preview,
            }
            X = pd.DataFrame([x]).reindex(columns=feature_order, fill_value=0.0)

            pred_total = float(reg.predict(X)[0])
            prob_over25 = float(clf.predict_proba(X)[0, 1])

            db.execute(text("""
                INSERT INTO predictions (fixture_id, predicted_total_goals, over_2_5_probability, updated_at)
                VALUES (:f, :ptg, :p25, NOW())
                ON CONFLICT (fixture_id)
                DO UPDATE SET
                  predicted_total_goals=EXCLUDED.predicted_total_goals,
                  over_2_5_probability=EXCLUDED.over_2_5_probability,
                  updated_at=NOW();
            """), {"f": r["fixture_id"], "ptg": pred_total, "p25": prob_over25})
            db.commit()
            log.info("prediction_upserted", extra={"fixture_id": r["fixture_id"]})
        except Exception as e:
            db.rollback()
            log.error(f"prediction_failed fixture={r['fixture_id']} err={e}")

if __name__ == "__main__":
    main()
```

---

# 7) Quick Notes / How to wire up

* **Dependencies** (typical):

  * `requests`, `SQLAlchemy`, `psycopg2-binary` (or `psycopg2`), `pandas`, `scikit-learn`, `xgboost`, `joblib`
* **Env**

  * `SOCCERDATA_API_KEY` must be set (token string)
  * `DATABASE_URL` to your Postgres instance
* **Run order**

  1. Apply SQL migration
  2. Train: `python scripts/train_model.py`
  3. Predict: `python standalone_predictor_outputter.py`

---