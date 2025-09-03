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
            log.debug("Sleeping for API rate limit", extra={
                "sleep_seconds": retry_after, "sleep_reason": "retry_after_header",
                "attempt": attempt + 1
            })
            time.sleep(retry_after)
        else:
            # exponential backoff starting ~1s
            sleep_duration = min(2 ** attempt, 16)
            log.debug("Sleeping for exponential backoff", extra={
                "sleep_seconds": sleep_duration, "sleep_reason": "exponential_backoff",
                "attempt": attempt + 1, "max_backoff": 16
            })
            time.sleep(sleep_duration)

    def _base_url(self, path: str) -> str:
        return f"{self.cfg.base_url.rstrip('/')}{path}"

    def _get(self, url: str, params: dict, etag: Optional[str] = None) -> requests.Response:
        p = dict(params or {})
        # The API examples pass auth_token as a query parameter
        p["auth_token"] = self.cfg.auth_token
        headers = {}
        if etag:
            headers["If-None-Match"] = etag
            log.debug("Using ETag for conditional GET", extra={
                "url": url, "etag": etag, "timeout": self.cfg.timeout
            })
        
        # Timeout usage: self.cfg.timeout applies to both connection and read operations
        # Default timeout prevents indefinite hangs on slow/unresponsive API endpoints
        log.debug("Making HTTP request", extra={
            "url": url, "params_count": len(p), "has_etag": bool(etag),
            "timeout_seconds": self.cfg.timeout, "timeout_usage": "connection_and_read"
        })
        
        return self.session.get(url, params=p, headers=headers, timeout=self.cfg.timeout)

    # ---------- H2H ----------
    def get_h2h_stats(self, team1_id: int, team2_id: int, competition_id: Optional[int] = None) -> Dict[str, Any]:
        """Return flattened H2H dict (includes advanced splits) using the new API format.
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
        url = "https://api.soccerdataapi.com/head-to-head/"
        params = {"team_1_id": team1_id, "team_2_id": team2_id}
        if competition_id:
            params["competition_id"] = competition_id

        response = None
        for attempt in range(self.cfg.max_retries + 1):
            log.debug("H2H request attempt", extra={
                "attempt": attempt + 1, "max_retries": self.cfg.max_retries + 1,
                "team1_id": team1_id, "team2_id": team2_id, "has_etag": bool(etag)
            })
            
            try:
                response = self._get(url, params, etag=etag)
                
                if response.status_code == 304 and row:
                    # Not modified → refresh timestamp only
                    log.debug("ETag match - refreshing cache timestamp", extra={
                        "team1_id": team1_id, "team2_id": team2_id, "etag": etag,
                        "cache_action": "timestamp_refresh"
                    })
                    self.db.execute(
                        text("""UPDATE h2h_cache SET last_fetched_at=NOW() 
                                WHERE team1_id=:t1 AND team2_id=:t2 AND COALESCE(competition_id,0)=:c"""),
                        {"t1": team1_id, "t2": team2_id, "c": key_comp}
                    )
                    self.db.commit()
                    log.debug("Cache timestamp updated successfully", extra={
                        "team1_id": team1_id, "team2_id": team2_id, "db_operation": "timestamp_update"
                    })
                    return dict(row)

                if response.status_code == 429:
                    ra = float(response.headers.get("Retry-After", "1"))
                    log.debug("Rate limit hit, sleeping", extra={
                        "attempt": attempt + 1, "retry_after_seconds": ra,
                        "status_code": 429, "retry_strategy": "rate_limit_header"
                    })
                    self._rate_limit_sleep(attempt, retry_after=ra)
                    continue

                if response.status_code >= 500:
                    log.debug("Server error, retrying with exponential backoff", extra={
                        "attempt": attempt + 1, "status_code": response.status_code,
                        "retry_strategy": "exponential_backoff"
                    })
                    self._rate_limit_sleep(attempt)
                    continue

                if response.status_code != 200:
                    log.error("H2H API error", extra={
                        "status_code": response.status_code, "response_text": response.text,
                        "team1_id": team1_id, "team2_id": team2_id
                    })
                    raise APIError(f"H2H HTTP {response.status_code}: {response.text}")

                log.debug("H2H API request successful", extra={
                    "attempt": attempt + 1, "status_code": response.status_code,
                    "response_size_bytes": len(response.content) if response.content else 0
                })
                data = response.json()
                break
                
            except requests.RequestException as e:
                log.debug("H2H request exception", extra={
                    "attempt": attempt + 1, "exception_type": type(e).__name__,
                    "exception_message": str(e), "will_retry": attempt < self.cfg.max_retries
                })
                if attempt >= self.cfg.max_retries:
                    log.error("H2H request failed after all retries", extra={
                        "total_attempts": attempt + 1, "final_exception": str(e)
                    })
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
            "competition_id": competition_id or 0,
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
        
        log.debug("Upserting H2H data to cache", extra={
            "team1_id": team1_id, "team2_id": team2_id, "competition_id": competition_id,
            "new_etag": etag_new, "games_played": overall_games_played,
            "avg_total_goals": avg_total_goals, "db_operation": "h2h_cache_upsert"
        })
        
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
            ON CONFLICT (team1_id, team2_id, competition_id)
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
        
        log.debug("H2H cache upsert completed successfully", extra={
            "team1_id": team1_id, "team2_id": team2_id, "db_operation": "commit_success"
        })

        return flat

    # ---------- Match Preview ----------
    def get_match_preview(self, match_id: int) -> Dict[str, Any]:
        """Return preview dict with excitement_rating and content list. Short TTL cached is handled by caller if desired."""
        url = self._base_url(self.cfg.preview_path)
        params = {"match_id": match_id}

        for attempt in range(self.cfg.max_retries + 1):
            log.debug("Match preview request attempt", extra={
                "attempt": attempt + 1, "max_retries": self.cfg.max_retries + 1,
                "match_id": match_id, "timeout_seconds": self.cfg.timeout
            })
            
            try:
                resp = self._get(url, params)
                
                if resp.status_code == 429:
                    ra = float(resp.headers.get("Retry-After", "1"))
                    log.debug("Preview rate limit hit, sleeping", extra={
                        "attempt": attempt + 1, "retry_after_seconds": ra,
                        "match_id": match_id, "retry_strategy": "rate_limit_header"
                    })
                    self._rate_limit_sleep(attempt, retry_after=ra)
                    continue
                    
                if resp.status_code >= 500:
                    log.debug("Preview server error, retrying", extra={
                        "attempt": attempt + 1, "status_code": resp.status_code,
                        "match_id": match_id, "retry_strategy": "exponential_backoff"
                    })
                    self._rate_limit_sleep(attempt)
                    continue
                    
                if resp.status_code != 200:
                    log.error("Preview API error", extra={
                        "status_code": resp.status_code, "response_text": resp.text,
                        "match_id": match_id
                    })
                    raise APIError(f"Preview HTTP {resp.status_code}: {resp.text}")
                    
                log.debug("Preview request successful", extra={
                    "attempt": attempt + 1, "status_code": resp.status_code,
                    "match_id": match_id, "response_size_bytes": len(resp.content) if resp.content else 0
                })
                return resp.json()
                
            except requests.RequestException as e:
                log.debug("Preview request exception", extra={
                    "attempt": attempt + 1, "exception_type": type(e).__name__,
                    "exception_message": str(e), "match_id": match_id,
                    "will_retry": attempt < self.cfg.max_retries
                })
                if attempt >= self.cfg.max_retries:
                    log.error("Preview request failed after all retries", extra={
                        "total_attempts": attempt + 1, "match_id": match_id,
                        "final_exception": str(e)
                    })
                    raise APIError(f"Preview request failed: {e}")
                self._rate_limit_sleep(attempt)