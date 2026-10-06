"""
College football win-probability model.

This module turns the exploratory notebook into a reusable weekly workflow.
The core model preserves the final approach developed in the notebook:

1. Historical team strength from talent, recruiting, returning production,
   previous-season point differential, and previous-season win percentage.
2. Preseason Ridge regression to estimate expected point differential.
3. Chronological dynamic ratings updated after each completed game.
4. Logistic regression mapping pregame rating difference + home field to win
   probability.
5. Monte Carlo simulation of the remaining schedule.

Data source:
SportsDataverse cfbfastR-cfb-data (public GitHub data).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from datetime import datetime
import urllib.request

import io
import certifi
import requests

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt 

CURRENT_SEASON = datetime.now().year

DEFAULT_SCHEDULE_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/"
    "releases/download/espn_cfb_schedules/"
    "cfb_schedule_{season}.parquet"
)

PRIMARY_STRENGTH_FEATURES = [
    "talent_composite",
    "avg_grade",
    "overall_returning",
    "previous_avg_point_diff",
    "previous_win_pct",
]

FALLBACK_STRENGTH_FEATURES = [
    "overall_returning",
    "previous_avg_point_diff",
    "previous_win_pct",
]

MINIMAL_FALLBACK_FEATURES = [
    "previous_avg_point_diff",
    "previous_win_pct",
]

PROBABILITY_FEATURES = [
    "pre_game_rating_difference",
    "home_field",
]


class CFBModel:
    """Reusable college-football model and weekly reporting interface."""

    def __init__(
        self,
        project_dir: str | Path | None = None,
        current_season: int = 2026,
        n_simulations: int = 5000,
        learning_rate: float = 0.15,
        home_field_advantage: float = 3.5,
        random_state: int | None = 42,
    ):
        self.project_dir = (
            Path(project_dir).resolve()
            if project_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        self.data_path = self.project_dir / "data" / "raw"
        self.current_season = current_season
        self.n_simulations = n_simulations
        self.learning_rate = learning_rate
        self.home_field_advantage = home_field_advantage
        self.random_state = random_state

        self.final_ratings: dict[tuple[int, str], float] = {}
        self.preseason_ratings: dict[tuple[int, str], float] = {}
        self.team_names: pd.DataFrame | None = None
        self.power_rankings: pd.DataFrame | None = None
        self.remaining_games: pd.DataFrame | None = None
        self.final_probability_model: LogisticRegression | None = None
        self.margin_error_mean: float = 0.0
        self.margin_error_std: float = 16.0
        self.team_ids: list[str] = []
        self.team_to_index: dict[str, int] = {}
        self.win_totals: np.ndarray | None = None
        self.last_update_summary: dict = {}

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    def refresh_current_schedule(self) -> Path:
        """Download the current-season schedule into data/raw."""
        self.data_path.mkdir(parents=True, exist_ok=True)

        url = DEFAULT_SCHEDULE_URL.format(
        season=self.current_season
        )

        response = requests.get(
        url,
        verify=certifi.where(),
        timeout=60
        )

        response.raise_for_status()

        schedule = pd.read_parquet(
        io.BytesIO(response.content)
        )

        destination = (
        self.data_path
        / f"cfb_schedules_{self.current_season}.csv"
        )

        schedule.to_csv(
        destination,
        index=False
        )

        return destination
    # ------------------------------------------------------------------
    # Public workflow
    # ------------------------------------------------------------------

    def update(self, refresh_schedule: bool = True) -> "CFBModel":
        """
        Rebuild the model from the raw files.

        This is the main weekly command. By default it refreshes the current
        schedule first, then rebuilds all historical features and ratings.
        """
        if refresh_schedule:
            self.refresh_current_schedule()

        self._load_raw_data()
        self._prepare_schedules()
        self._build_team_features()
        self._build_preseason_strength()
        self._build_dynamic_ratings()
        self._fit_probability_model()
        self._build_remaining_schedule()
        self._calibrate_margin_error()
        self._build_power_rankings()
        

        self.last_update_summary = {
            "season": self.current_season,
            "completed_fbs_games": len(self.fbs_games),
            "remaining_fbs_games": len(self.remaining_games),
            "teams_with_current_ratings": len(
                self.final_ratings_for_current_season
            ),
        }

        return self

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_raw_data(self):
        schedules = []
        talents = []
        recruiting = []
        returning = []

        schedule_files = sorted(
            self.data_path.glob("cfb_schedules_*.csv"),
            key=self._season_from_filename,
        )
        talent_files = sorted(
            self.data_path.glob("cfb_team_talent_*.parquet"),
            key=self._season_from_filename,
        )
        recruiting_files = sorted(
            self.data_path.glob("cfb_recruits_*.parquet"),
            key=self._season_from_filename,
        )
        returning_files = sorted(
            self.data_path.glob("cfb_returning_production_*.parquet"),
            key=self._season_from_filename,
        )

        if not schedule_files:
            raise FileNotFoundError(
                f"No schedule CSVs found in {self.data_path}"
            )
        if not talent_files:
            raise FileNotFoundError(
                f"No team-talent parquet files found in {self.data_path}"
            )
        if not recruiting_files:
            raise FileNotFoundError(
                f"No recruiting parquet files found in {self.data_path}"
            )
        if not returning_files:
            raise FileNotFoundError(
                f"No returning-production parquet files found in {self.data_path}"
            )

        for file in schedule_files:
            schedules.append(pd.read_csv(file))

        for file in talent_files:
            talents.append(pd.read_parquet(file))

        for file in recruiting_files:
            recruiting.append(pd.read_parquet(file))

        for file in returning_files:
            returning.append(pd.read_parquet(file))

        self.all_schedules = pd.concat(
            schedules,
            ignore_index=True,
            sort=False,
        )

        self.all_talent = pd.concat(
            talents,
            ignore_index=True,
            sort=False,
        )

        self.all_recruiting = pd.concat(
            recruiting,
            ignore_index=True,
            sort=False,
        )

        self.all_returning_production = pd.concat(
            returning,
            ignore_index=True,
            sort=False,
        )

        self.all_talent["team_id"] = (
            self.all_talent["team_id"].astype(str)
        )

        self.all_recruiting["team_id"] = (
            self.all_recruiting["team_id"].astype(str)
        )

        self.all_returning_production["team_id"] = (
            self.all_returning_production["team_id"].astype(str)
        )
       
        

        self.recruiting_summary = (
            self.all_recruiting
            .groupby(["season", "team_id", "team"])
            .agg(
                recruits=("recruit_id", "count"),
                avg_stars=("stars", "mean"),
                avg_grade=("grade", "mean"),
                four_star_recruits=(
                    "stars",
                    lambda x: (x == 4).sum(),
                ),
                five_star_recruits=(
                    "stars",
                    lambda x: (x == 5).sum(),
                ),
            )
            .reset_index()
        )

        self.team_season = self.all_talent.merge(
            self.recruiting_summary,
            on=["season", "team_id", "team"],
            how="left",
        )

        self.team_season = self.team_season.merge(
            self.all_returning_production,
            on=["season", "team_id"],
            how="left",
            suffixes=("", "_returning"),
        )
       

    @staticmethod
    def _season_from_filename(path: Path):
        match = path.stem.rsplit("_", 1)[-1]

        try:
            return int(match)
        except ValueError:
            return 9999

    # ------------------------------------------------------------------
    # Schedule preparation
    # ------------------------------------------------------------------

    def _prepare_schedules(self):
        standardized = []

        schedule_files = []

    # Historical schedules: cfb_schedule_YYYY.csv
        schedule_files.extend(
        self.data_path.glob("cfb_schedule_*.csv")
        )

    # Current schedule: cfb_schedules_YYYY.csv
        current_file = self.data_path / f"cfb_schedules_{self.current_season}.csv"

        if current_file.exists():
            schedule_files.append(current_file)

    # Remove duplicates just in case
        schedule_files = sorted(
        set(schedule_files),
        key=self._season_from_filename,
        )

        
        for file in schedule_files:
            temp = pd.read_csv(file)
           

            # Older 2021-2025 format.
            if "home_score" in temp.columns:
                temp = temp.rename(
                columns={
                    "game_date": "start_date",
                    "home_score": "home_points",
                    "away_score": "away_points",
                }
                )

            required = [
            "game_id",
            "season",
            "week",
            "season_type",
            "start_date",
            "neutral_site",
            "home_id",
            "away_id",
            "home_team",
            "away_team",
            "home_points",
            "away_points",
            "home_winner",
            "away_winner",
            "status",
            ]

            missing = [column for column in required if column not in temp.columns]

            if missing:
                raise ValueError(
                f"{file.name} is missing columns: {missing}"
                )

            standardized.append(temp[required].copy())

        self.all_games = pd.concat(
        standardized,
        ignore_index=True,
        )
    
        
        self.all_games["start_date"] = pd.to_datetime(
            self.all_games["start_date"],
            format="mixed",
            utc=True,
        )

        self.all_games["home_id"] = (
        pd.to_numeric(self.all_games["home_id"], errors="coerce")
        .astype("Int64")
        .astype(str)
        )

        self.all_games["away_id"] = (
        pd.to_numeric(self.all_games["away_id"], errors="coerce")
        .astype("Int64")
        .astype(str)
        )

        self.all_games = self.all_games.dropna(
            subset=[
                "home_points",
                "away_points",
            ]
        ).copy()

        self.all_games = (
            self.all_games
            .sort_values("start_date")
            .reset_index(drop=True)
        )

        # Current-season schedule supplies the FBS team set.
        current_file = (
            self.data_path
            / f"cfb_schedules_{self.current_season}.csv"
        )

        current_schedule = pd.read_csv(current_file)

        

        current_fbs = current_schedule.copy()

        current_fbs["home_id"] = (
        pd.to_numeric(current_fbs["home_id"], errors="coerce")
        .astype("Int64")
        .astype(str)
        )

        current_fbs["away_id"] = (
        pd.to_numeric(current_fbs["away_id"], errors="coerce")
        .astype("Int64")
        .astype(str)
        )

        self.fbs_teams = set(
        pd.concat(
        [
            current_fbs["home_id"],
            current_fbs["away_id"],
        ]
        )
        .dropna()
        .astype(str)
        )       

        

        self.regular_games = self.all_games[
        self.all_games["season_type"] == 2
        ].copy()

        

        self.fbs_games = self.regular_games[
            self.regular_games["home_id"].isin(self.fbs_teams)
            & self.regular_games["away_id"].isin(self.fbs_teams)
        ].copy()
       
        self.fbs_games["home_win"] = (
            self.fbs_games["home_points"]
            > self.fbs_games["away_points"]
        ).astype(int)

        self.fbs_games = (
            self.fbs_games
            .sort_values("start_date")
            .reset_index(drop=True)
        )

        self.schedule_current = current_schedule

        self.completed_current = self.fbs_games[
            self.fbs_games["season"] == self.current_season
        ].copy()

    # ------------------------------------------------------------------
    # Team-season features
    # ------------------------------------------------------------------

    def _build_team_features(self):
       
        home_results = self.fbs_games[
            [
                "season",
                "start_date",
                "home_id",
                "home_points",
                "away_points",
            ]
        ].copy()

        home_results = home_results.rename(
            columns={
                "home_id": "team_id",
            }
        )

        home_results["point_diff"] = (
            home_results["home_points"]
            - home_results["away_points"]
        )

        away_results = self.fbs_games[
            [
                "season",
                "start_date",
                "away_id",
                "away_points",
                "home_points",
            ]
        ].copy()

        away_results = away_results.rename(
            columns={
                "away_id": "team_id",
            }
        )

        away_results["point_diff"] = (
            away_results["away_points"]
            - away_results["home_points"]
        )

        self.team_results = pd.concat(
            [
                home_results,
                away_results,
            ],
            ignore_index=True,
        )

        self.team_results["team_id"] = (
            self.team_results["team_id"].astype(str)
        )

        self.team_results = (
            self.team_results
            .sort_values(
                ["team_id", "start_date"]
            )
            .reset_index(drop=True)
        )

        season_avg = (
            self.team_results
            .groupby(
                ["team_id", "season"]
            )["point_diff"]
            .mean()
            .reset_index(
                name="season_avg_diff"
            )
            .sort_values(
                ["team_id", "season"]
            )
        )

        season_avg["previous_season_diff"] = (
            season_avg
            .groupby("team_id")["season_avg_diff"]
            .shift(1)
        )

        self.team_results = self.team_results.merge(
            season_avg[
                [
                    "team_id",
                    "season",
                    "previous_season_diff",
                ]
            ],
            on=[
                "team_id",
                "season",
            ],
            how="left",
        )
        self.team_results["prior_avg_diff"] = (
            self.team_results
            .groupby(["team_id", "season"])["point_diff"]
            .transform(lambda x: x.shift().expanding().mean())
        )

        self.team_results["prior_avg_diff"] = (
            self.team_results["prior_avg_diff"]
            .fillna(self.team_results["previous_season_diff"])
        )

        self.team_results["win"] = (
            self.team_results["point_diff"] > 0
        ).astype(int)

        self.team_performance = (
            self.team_results
            .groupby(["team_id", "season"])
            .agg(
                avg_point_diff=("point_diff", "mean"),
                games=("point_diff", "count"),
                win_pct=("win", "mean"),
            )
            .reset_index()
            .sort_values(["team_id", "season"])
        )

        self.team_performance["previous_avg_point_diff"] = (
            self.team_performance
            .groupby("team_id")["avg_point_diff"]
            .shift(1)
        )

        self.team_performance["previous_win_pct"] = (
            self.team_performance
            .groupby("team_id")["win_pct"]
            .shift(1))
       

        self.team_season["team_id"] = self.team_season["team_id"].astype(str)

        self.team_season = self.team_season.merge(
            self.team_performance[
                [
                    "team_id",
                    "season",
                    "previous_avg_point_diff",
                    "previous_win_pct",
                    "avg_point_diff",
                ]
            ],
            on=["team_id", "season"],
            how="left",
            suffixes=("", "_performance"),
        )
       

    # ------------------------------------------------------------------
    # Preseason strength
    # ------------------------------------------------------------------

    def _build_preseason_strength(self):
        strength_data = self.team_season.dropna(
            subset=PRIMARY_STRENGTH_FEATURES + ["avg_point_diff"]
        ).copy()
        

        self.preseason_models = {}

        preseason_results = []

        for season in range(2023, self.current_season + 1):
            train = strength_data[
                strength_data["season"] < season
            ].copy()

            current = self.team_season[
                self.team_season["season"] == season
            ].copy()

            

            model = Ridge(alpha=1.0)

            model.fit(
                train[PRIMARY_STRENGTH_FEATURES],
                train["avg_point_diff"],
            )

            self.preseason_models[season] = model

            current["preseason_strength"] = np.nan

            complete = (
                current[PRIMARY_STRENGTH_FEATURES]
                .notna()
                .all(axis=1)
            )

            if complete.any():
                current.loc[
                    complete,
                    "preseason_strength",
                ] = model.predict(
                    current.loc[
                        complete,
                        PRIMARY_STRENGTH_FEATURES,
                    ]
                )

            # ----------------------------------------------------------
            # First fallback model
            # ----------------------------------------------------------

            fallback_train = self.team_season.merge(
                self.team_performance[
                    [
                        "team_id",
                        "season",
                        "avg_point_diff",
                    ]
                ],
                on=[
                    "team_id",
                    "season",
                ],
                how="left",
                suffixes=("", "_perf"),
            )

            if "avg_point_diff_perf" in fallback_train.columns:
                fallback_target = "avg_point_diff_perf"
            else:
                fallback_target = "avg_point_diff"

            fallback_train = fallback_train.dropna(
                subset=FALLBACK_STRENGTH_FEATURES
                + [fallback_target]
            )

            fallback = Ridge(alpha=1.0)

            fallback.fit(
                fallback_train[FALLBACK_STRENGTH_FEATURES],
                fallback_train[fallback_target],
            )

            need = current["preseason_strength"].isna()

            fallback_complete = (
                current[FALLBACK_STRENGTH_FEATURES]
                .notna()
                .all(axis=1)
            )

            use_fallback = (
                need
                & fallback_complete
            )

            if use_fallback.any():
                current.loc[
                    use_fallback,
                    "preseason_strength",
                ] = fallback.predict(
                    current.loc[
                        use_fallback,
                        FALLBACK_STRENGTH_FEATURES,
                    ]
                )

            # ----------------------------------------------------------
            # Second fallback model
            # ----------------------------------------------------------

            minimal_train = self.team_season.merge(
                self.team_performance[
                    [
                        "team_id",
                        "season",
                        "avg_point_diff",
                    ]
                ],
                on=[
                    "team_id",
                    "season",
                ],
                how="left",
                suffixes=("", "_perf2"),
            )

            target2 = (
                "avg_point_diff_perf2"
                if "avg_point_diff_perf2"
                in minimal_train.columns
                else "avg_point_diff"
            )

            minimal_train = minimal_train.dropna(
                subset=MINIMAL_FALLBACK_FEATURES
                + [target2]
            )

            minimal = Ridge(alpha=1.0)

            minimal.fit(
                minimal_train[MINIMAL_FALLBACK_FEATURES],
                minimal_train[target2],
            )

            need = current["preseason_strength"].isna()

            minimal_complete = (
                current[MINIMAL_FALLBACK_FEATURES]
                .notna()
                .all(axis=1)
            )

            use_minimal = (
                need
                & minimal_complete
            )

            if use_minimal.any():
                current.loc[
                    use_minimal,
                    "preseason_strength",
                ] = minimal.predict(
                    current.loc[
                        use_minimal,
                        MINIMAL_FALLBACK_FEATURES,
                    ]
                )

            # Final safety net.
            current["preseason_strength"] = (
                current["preseason_strength"]
                .fillna(
                    strength_data["avg_point_diff"].median()
                )
            )

            preseason_results.append(
                current[
                    [
                        "team_id",
                        "season",
                        "preseason_strength",
                    ]
                ]
            )

        self.all_preseason_strength = pd.concat(
            preseason_results,
            ignore_index=True,
        )

        self.all_preseason_strength["team_id"] = (
            self.all_preseason_strength["team_id"]
            .astype(str)
        )

    # ------------------------------------------------------------------
    # Dynamic ratings
    # ------------------------------------------------------------------

    def _build_dynamic_ratings(self):
        dynamic_games = (
            self.fbs_games[
                self.fbs_games["season"].between(
                    2023,
                    self.current_season,
                )
            ]
            .sort_values("start_date")
            .reset_index(drop=True)
        )
       

        ratings = {
            (
                int(row["season"]),
                str(row["team_id"]),
            ): float(row["preseason_strength"])
            for _, row
            in self.all_preseason_strength.iterrows()
        }

        results = []

        for _, game in dynamic_games.iterrows():
            season = int(game["season"])

            home_id = str(game["home_id"])
            away_id = str(game["away_id"])

            home_key = (
                season,
                home_id,
            )

            away_key = (
                season,
                away_id,
            )

            if (
                home_key not in ratings
                or away_key not in ratings
            ):
                continue

            home_rating = ratings[home_key]
            away_rating = ratings[away_key]

            home_field = int(
                not bool(game["neutral_site"])
            )

            expected_margin = (
                home_rating
                - away_rating
                + self.home_field_advantage
                * home_field
            )

            actual_margin = (
                float(game["home_points"])
                - float(game["away_points"])
            )

            results.append(
                {
                    "season": season,
                    "start_date": game["start_date"],
                    "home_id": home_id,
                    "away_id": away_id,
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "home_win": int(
                        game["home_win"]
                    ),
                    "pre_game_rating_difference": (
                        home_rating
                        - away_rating
                    ),
                    "home_field": home_field,
                    "expected_margin": expected_margin,
                    "actual_margin": actual_margin,
                }
            )

            error = (
                actual_margin
                - expected_margin
            )

            ratings[home_key] += (
                self.learning_rate
                * error
            )

            ratings[away_key] -= (
                self.learning_rate
                * error
            )

        self.final_dynamic_results = (
            pd.DataFrame(results)
        )
        
        self.final_ratings = ratings

        self.final_ratings_for_current_season = {
            key: value
            for key, value in ratings.items()
            if key[0] == self.current_season
        }

        self.preseason_ratings = {
            (
                int(row["season"]),
                str(row["team_id"]),
            ): float(
                row["preseason_strength"]
            )
            for _, row
            in self.all_preseason_strength.iterrows()
        }

    # ------------------------------------------------------------------
    # Probability model
    # ------------------------------------------------------------------

    def _fit_probability_model(self):
        train = self.final_dynamic_results[
            self.final_dynamic_results["season"]
            < self.current_season
        ].copy()

        self.final_probability_model = (
            LogisticRegression()
        )

        self.final_probability_model.fit(
            train[PROBABILITY_FEATURES],
            train["home_win"],
        )
        

        current = self.final_dynamic_results[
            self.final_dynamic_results["season"]
            == self.current_season
        ].copy()

        if len(current):
            probabilities = (
                self.final_probability_model
                .predict_proba(
                    current[
                        PROBABILITY_FEATURES
                    ]
                )[:, 1]
            )

            current[
                "home_win_probability"
            ] = probabilities

            self.current_evaluation = current

        else:
            self.current_evaluation = (
                pd.DataFrame()
            )

        historical = train.copy()

        historical[
            "home_win_probability"
        ] = (
            self.final_probability_model
            .predict_proba(
                historical[
                    PROBABILITY_FEATURES
                ]
            )[:, 1]
        )

        self.historical_evaluation = historical

    def _calibrate_margin_error(self):
        historical = (
            self.final_dynamic_results.copy()
        )

        errors = (
            historical["actual_margin"]
            - historical["expected_margin"]
        )

        self.margin_error_mean = float(
            errors.mean()
        )

        self.margin_error_std = float(
            errors.std()
        )

        if (
            not np.isfinite(
                self.margin_error_std
            )
            or self.margin_error_std <= 0
        ):
            self.margin_error_std = 16.0

    # ------------------------------------------------------------------
    # Remaining schedule and rankings
    # ------------------------------------------------------------------

    def _build_remaining_schedule(self):
                

        remaining = self.schedule_current[
        self.schedule_current["status"].astype(str).str.upper() != "STATUS_FINAL"
        ].copy()

        remaining["home_id"] = (
            remaining["home_id"]
            .astype(str)
        )

        remaining["away_id"] = (
            remaining["away_id"]
            .astype(str)
        )

        remaining = remaining[
            remaining["home_id"].isin(
                self.fbs_teams
            )
            &
            remaining["away_id"].isin(
                self.fbs_teams
            )
        ].copy()

        remaining = (
            remaining
            .sort_values(
                [
                    "game_date",
                    "week",
                ]
            )
            .reset_index(drop=True)
        )

        self.remaining_games = remaining

        # Build a team-name lookup from the current schedule.
        names = pd.concat(
            [
                self.schedule_current[
                    [
                        "home_id",
                        "home_team",
                    ]
                ].rename(
                    columns={
                        "home_id": "team_id",
                        "home_team": "team",
                    }
                ),
                self.schedule_current[
                    [
                        "away_id",
                        "away_team",
                    ]
                ].rename(
                    columns={
                        "away_id": "team_id",
                        "away_team": "team",
                    }
                ),
            ],
            ignore_index=True,
        )

        names["team_id"] = (
            names["team_id"].astype(str)
        )

        self.team_names = (
            names.drop_duplicates(
                "team_id"
            )
        )

        # Ensure every current-season FBS team
        # has a rating.
        current_ids = sorted(
            self.fbs_teams
        )

        median_rating = float(
            np.median(
                list(
                    self.final_ratings_for_current_season.values()
                )
            )
        )

        for team_id in current_ids:
            key = (
                self.current_season,
                team_id,
            )

            if (
                key
                not in self.final_ratings_for_current_season
            ):
                self.final_ratings[key] = (
                    median_rating
                )

                self.final_ratings_for_current_season[
                    key
                ] = median_rating

    def _build_power_rankings(self):
        rows = []

        for key, rating in (
            self.final_ratings_for_current_season.items()
        ):
            rows.append(
                {
                    "team_id": key[1],
                    "team": None,
                    "current_rating": rating,
                }
            )

        rankings = pd.DataFrame(rows)

        rankings = rankings.merge(
            self.team_names,
            on="team_id",
            how="left",
        )

        rankings = (
            rankings
            .sort_values(
                "current_rating",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        rankings["rank"] = (
            np.arange(
                1,
                len(rankings) + 1,
            )
        )

        self.power_rankings = rankings[
            [
                "rank",
                "team_y",
                "team_id",
                "current_rating",
            ]
        ].copy()

        self.power_rankings = self.power_rankings.rename(columns={"team_y":"team"})

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def top25(self) -> pd.DataFrame:
        """Return the current dynamic Top 25."""
        self._require_updated()

        top25 = self.power_rankings.head(
            25
        ).copy()

        plt.figure(figsize=(9, 10))

        plt.barh(
            top25["team"][::-1],
            top25["current_rating"][::-1],
        )

        plt.xlabel("Strength Rating")
        plt.ylabel("Team")
        plt.title(
            f"Top 25 College Football Teams by Model Rating "
            f"({self.current_season})"
        )

        plt.grid(
            axis="x",
            alpha=0.3,
        )

        plt.tight_layout()

        plt.savefig(
            "outputs/figures/top25_ratings.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()

        return top25

    def team(
        self,
        team_name: str,
    ) -> pd.DataFrame:
        """
        Return all remaining games for a team
        with model win probabilities.
        """
        self._require_updated()

        team_id = self._resolve_team(
            team_name
        )
        remaining = self.remaining_games[
            (self.remaining_games["home_id"] == team_id)
            | (self.remaining_games["away_id"] == team_id)
        ].copy()

        if remaining.empty:
            return remaining

        rows = []

        for _, game in remaining.iterrows():
            home_id = str(game["home_id"])
            away_id = str(game["away_id"])

            home_rating = self.final_ratings_for_current_season.get(
                (self.current_season, home_id)
            )

            away_rating = self.final_ratings_for_current_season.get(
                (self.current_season, away_id)
            )

            if home_rating is None or away_rating is None:
                continue

            home_field = int(
                not bool(game["neutral_site"])
            )

            rating_difference = (
                home_rating - away_rating
            )

            probability = self._home_win_probability(
                rating_difference,
                home_field,
            )

            if home_id == team_id:
                team_probability = probability
                opponent = game["away_team"]
                team_is_home = True
            else:
                team_probability = 1 - probability
                opponent = game["home_team"]
                team_is_home = False

            rows.append(
                {
                    "date": game["start_date"],
                    "week": game["week"],
                    "team": team_name,
                    "opponent": opponent,
                    "home": team_is_home,
                    "win_probability": team_probability,
                }
            )

        return pd.DataFrame(rows)



    def info(
        self,
        team_name: str,
    ) -> pd.DataFrame:
        """Return current-season model information for a team."""
        self._require_updated()

        team_id = self._resolve_team(team_name)

        current_fbs_teams = set(
            pd.concat(
                [
                    self.fbs_games.loc[
                        self.fbs_games["season"] == self.current_season,
                        "home_id",
                    ],
                    self.fbs_games.loc[
                        self.fbs_games["season"] == self.current_season,
                        "away_id",
                    ],
                ]
            )
            .dropna()
            .astype(str)
        )

        current_ratings = {
            team_id: rating
            for (season, team_id), rating
            in self.final_ratings_for_current_season.items()
            if (
                season == self.current_season
                and team_id in current_fbs_teams
            )
        }

        if (self.current_season, team_id) not in (
            self.final_ratings_for_current_season
        ):
            return pd.DataFrame()

        team_rating = current_ratings[team_id]

        rankings = (
            pd.Series(current_ratings)
            .sort_values(ascending=False)
            .reset_index()
        )

        rankings.columns = [
            "team_id",
            "strength_rating",
        ]

        rankings["rank"] = (
            rankings["strength_rating"]
            .rank(
                method="min",
                ascending=False,
            )
            .astype(int)
        )

        team_rank = rankings.loc[
            rankings["team_id"] == team_id,
            "rank",
        ].iloc[0]

        return pd.DataFrame([{
            "team": team_name,
            "model_rank": f"{team_rank} / {len(rankings)}",
            "strength_rating": team_rating,
        }])
    

    def game(
        self,
        team_a: str,
        team_b: str,
    ) -> pd.DataFrame:
        """Return the model probability for a hypothetical matchup."""
        self._require_updated()

        team_a_id = self._resolve_team(team_a)
        team_b_id = self._resolve_team(team_b)

        # Use the current-season ratings.
        team_a_rating = self.final_ratings_for_current_season[
        (self.current_season, team_a_id)
        ]

        team_b_rating = self.final_ratings_for_current_season[
        (self.current_season, team_b_id)
        ]

        # Treat team_a as the home team for the hypothetical matchup.
        home_field = 1

        rating_difference = (
        team_a_rating - team_b_rating
        )

        team_a_probability = self._home_win_probability(
        rating_difference,
        home_field,
        )

        return pd.DataFrame([{
        "team": team_a,
        "opponent": team_b,
        "team_rating": team_a_rating,
        "opponent_rating": team_b_rating,
        "team_win_probability": team_a_probability,
        "opponent_win_probability": 1 - team_a_probability,
        }])
    
    def evaluate(self) -> pd.DataFrame:
        """Evaluate the probability model on the current season."""
        self._require_updated()

        test = self.final_dynamic_results[
            self.final_dynamic_results["season"]
            == self.current_season
        ].copy()

        X_test = test[PROBABILITY_FEATURES]
        y_test = test["home_win"]

        probabilities = self.final_probability_model.predict_proba(
            X_test
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        log_loss_value = log_loss(
            y_test,
            probabilities,
        )

        brier_score_value = brier_score_loss(
            y_test,
            probabilities,
        )

        accuracy_value = (
            predictions == y_test
        ).mean()

        # Baseline: always predict the historical home-win rate
        baseline_probability = (
            self.final_dynamic_results[
            self.final_dynamic_results["season"]
                < self.current_season
            ]["home_win"]
            .mean()
        )

        baseline_probabilities = np.full(
            len(y_test),
            baseline_probability,
        )

        baseline_predictions = np.ones(
            len(y_test),
            dtype=int,
        )

        baseline_log_loss = log_loss(
            y_test,
            baseline_probabilities,
        )

        baseline_brier_score = brier_score_loss(
            y_test,
            baseline_probabilities,
        )

        baseline_accuracy = (
            baseline_predictions == y_test
        ).mean()




        # Performance comparison figure
        metrics = [
            "Accuracy",
            "Log Loss",
            "Brier Score",
        ]

        model_values = [
            accuracy_value,
            log_loss_value,
            brier_score_value,
        ]

        baseline_values = [
            baseline_accuracy,
            baseline_log_loss,
            baseline_brier_score,
        ]

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(12, 4),
        )

        # Accuracy
        axes[0].bar(
            ["Model", "Baseline"],
            [model_values[0], baseline_values[0]],
        )
        axes[0].set_title("Accuracy")
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Score")

        # Log loss
        axes[1].bar(
            ["Model", "Baseline"],
            [model_values[1], baseline_values[1]],
        )
        axes[1].set_title("Log Loss")
        axes[1].set_ylabel("Score")

        # Brier score
        axes[2].bar(
            ["Model", "Baseline"],
            [model_values[2], baseline_values[2]],
        )
        axes[2].set_title("Brier Score")
        axes[2].set_ylabel("Score")

        fig.suptitle(
            "Model Performance vs. Baseline",
            fontsize=14,
        )

        plt.tight_layout()

        plt.savefig(
            "outputs/figures/model_performance.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.show() 
 



        return pd.DataFrame([{
            "games_evaluated": len(test),
            "model_log_loss": log_loss_value,
            "baseline_log_loss": baseline_log_loss,
            "model_brier_score": brier_score_value,
            "baseline_brier_score": baseline_brier_score,
            "model_accuracy": accuracy_value,
            "baseline_accuracy": baseline_accuracy,
        }])



    def calibration(self) -> pd.DataFrame:
        """Check calibration of the probability model."""
        self._require_updated()

        test = self.final_dynamic_results[
            self.final_dynamic_results["season"]
            == self.current_season
        ].copy()

        X_test = test[PROBABILITY_FEATURES]
        y_test = test["home_win"]

        probabilities = self.final_probability_model.predict_proba(
            X_test
        )[:, 1]

        actual_rate, predicted_rate = calibration_curve(
            y_test,
            probabilities,
            n_bins=10,
            strategy="uniform",
        )




        plt.figure(figsize=(8, 6))

        plt.plot(
            predicted_rate,
            actual_rate,
            marker="o",
            linewidth=2,
            label="Model",
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1.5,
            label="Perfect calibration",
        )

        plt.xlabel("Predicted win probability")
        plt.ylabel("Actual win rate")
        plt.title("College Football Win Probability Calibration")

        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            "outputs/figures/calibration.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()
    


        return pd.DataFrame({
            "predicted_probability": predicted_rate,
            "actual_win_rate": actual_rate,
        })


    
    def simulate_team(
        self,
        team_name: str,
        n_simulations: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Simulate the remainder of the season and return the distribution
        of additional wins for the selected team.
        """
        self._require_updated()

        if n_simulations is None:
            n_simulations = self.n_simulations

        team_id = self._resolve_team(team_name)

        results = self._run_monte_carlo(
            n_simulations=n_simulations
        )

        team_index = self.team_to_index[team_id]

        remaining_wins = (
            results[:, team_index]
        )

        distribution = (
            pd.Series(remaining_wins)
            .value_counts()
            .sort_index()
            .reset_index()
        )

        distribution.columns = [
            "remaining_wins",
            "simulations",
        ]

        distribution["probability"] = (
            distribution["simulations"]
            / n_simulations
        )

        distribution["team"] = team_name

        return distribution[
            [
                "team",
                "remaining_wins",
                "simulations",
                "probability",
            ]
        ]

    def all_simulated_results(
        self,
        n_simulations: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Return simulated additional-win distributions for every team.
        """
        self._require_updated()

        if n_simulations is None:
            n_simulations = self.n_simulations

        results = self._run_monte_carlo(
            n_simulations=n_simulations
        )

        rows = []

        for team_id, index in self.team_to_index.items():
            values = results[:, index]

            for wins in np.unique(values):
                count = int(
                    np.sum(values == wins)
                )

                rows.append(
                    {
                        "team_id": team_id,
                        "team": self._team_name(team_id),
                        "remaining_wins": int(wins),
                        "simulations": count,
                        "probability": (
                            count / n_simulations
                        ),
                    }
                )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------

    def _run_monte_carlo(
        self,
        n_simulations: int,
    ) -> np.ndarray:
        """Simulate the remaining schedule."""
        remaining = self.remaining_games

        if remaining.empty:
            return np.zeros(
                (
                    n_simulations,
                    len(self.team_ids),
                ),
                dtype=np.int16,
            )

        home_ids = (
            remaining["home_id"]
            .astype(str)
            .to_numpy()
        )

        away_ids = (
            remaining["away_id"]
            .astype(str)
            .to_numpy()
        )

        home_field = (
            ~remaining["neutral_site"]
            .fillna(False)
        ).astype(int).to_numpy()

        team_ids = sorted(
            set(home_ids).union(
                set(away_ids)
            )
        )

        self.team_ids = team_ids

        self.team_to_index = {
            team_id: i
            for i, team_id
            in enumerate(team_ids)
        }

        home_indices = np.array(
            [
                self.team_to_index[x]
                for x in home_ids
            ]
        )

        away_indices = np.array(
            [
                self.team_to_index[x]
                for x in away_ids
            ]
        )

        starting_ratings = np.array(
            [
                self.final_ratings_for_current_season[
                    (self.current_season, team_id)
                ]
                for team_id in team_ids
            ]
        )

        win_totals = np.zeros(
            (
                n_simulations,
                len(team_ids),
            ),
            dtype=np.int16,
        )

        rng = np.random.default_rng(
            self.random_state
        )

        intercept = (
            self.final_probability_model
            .intercept_[0]
        )

        rating_coefficient = (
            self.final_probability_model
            .coef_[0][0]
        )

        home_field_coefficient = (
            self.final_probability_model
            .coef_[0][1]
        )

        for _ in range(n_simulations):
            ratings = (
                starting_ratings.copy()
            )

            for game_index in range(
                len(remaining)
            ):
                h = home_indices[
                    game_index
                ]

                a = away_indices[
                    game_index
                ]

                rating_difference = (
                    ratings[h]
                    - ratings[a]
                )

                logit = (
                    intercept
                    + rating_coefficient
                    * rating_difference
                    + home_field_coefficient
                    * home_field[
                        game_index
                    ]
                )

                home_win_probability = (
                    1
                    / (
                        1
                        + np.exp(-logit)
                    )
                )

                home_wins = (
                    rng.random()
                    < home_win_probability
                )

                expected_margin = (
                    rating_difference
                    + self.home_field_advantage
                    * home_field[
                        game_index
                    ]
                )

                simulated_margin = (
                    expected_margin
                    + rng.normal(
                        self.margin_error_mean,
                        self.margin_error_std,
                    )
                )

                if home_wins:
                    if simulated_margin <= 0:
                        simulated_margin = (
                            abs(
                                simulated_margin
                            )
                            + 0.1
                        )

                    win_totals[
                        _,
                        h,
                    ] += 1

                else:
                    if simulated_margin >= 0:
                        simulated_margin = (
                            -abs(
                                simulated_margin
                            )
                            - 0.1
                        )

                    win_totals[
                        _,
                        a,
                    ] += 1

                rating_error = (
                    simulated_margin
                    - expected_margin
                )

                ratings[h] += (
                    self.learning_rate
                    * rating_error
                )

                ratings[a] -= (
                    self.learning_rate
                    * rating_error
                )

        self.win_totals = win_totals

        return win_totals

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _home_win_probability(
        self,
        rating_difference: float,
        home_field: int,
    ) -> float:
        """Calculate home-team win probability."""
        if self.final_probability_model is None:
            raise RuntimeError(
                "Probability model has not been fitted."
            )

        intercept = (
            self.final_probability_model
            .intercept_[0]
        )

        rating_coefficient = (
            self.final_probability_model
            .coef_[0][0]
        )

        home_field_coefficient = (
            self.final_probability_model
            .coef_[0][1]
        )

        logit = (
            intercept
            + rating_coefficient
            * rating_difference
            + home_field_coefficient
            * home_field
        )

        return float(
            1
            / (
                1
                + np.exp(-logit)
            )
        )

    def _resolve_team(
        self,
        team_name: str,
    ) -> str:
        """Resolve a team name to its team ID."""
        query = team_name.strip().lower()

        names = self.team_names.copy()

        names["team_lower"] = (
            names["team"]
            .astype(str)
            .str.lower()
        )

        exact = names[
            names["team_lower"] == query
        ]

        if len(exact) == 1:
            return str(
                exact.iloc[0]["team_id"]
            )

        contains = names[
            names["team_lower"].str.contains(
                query,
                regex=False,
                na=False,
            )
        ]

        if len(contains) == 1:
            return str(
                contains.iloc[0]["team_id"]
            )

        if len(contains) > 1:
            matches = (
                contains["team"]
                .astype(str)
                .tolist()
            )

            raise ValueError(
                f"Multiple teams matched "
                f"'{team_name}': {matches}"
            )

        raise ValueError(
            f"Could not find team '{team_name}'."
        )

    def _team_name(
        self,
        team_id: str,
    ) -> str:
        """Return a team's display name."""
        match = self.team_names[
            self.team_names["team_id"]
            == str(team_id)
        ]

        if match.empty:
            return str(team_id)

        return str(
            match.iloc[0]["team"]
        )

    def _require_updated(self):
        """Make sure update() has been run."""
        if (
            self.remaining_games is None
            or self.final_probability_model is None
            or self.power_rankings is None
        ):
            raise RuntimeError(
                "Run model.update() before using "
                "this method."
            )

        
    def check_team(self, team_name):
        team_name = team_name.lower()

        games = self.fbs_games[
        (
            self.fbs_games["home_team"]
            .astype(str)
            .str.lower()
            .str.contains(team_name, na=False)
        )
        |
        (
            self.fbs_games["away_team"]
            .astype(str)
            .str.lower()
            .str.contains(team_name, na=False)
        )
        ].copy()

        if games.empty:
            return f"No games found for {team_name}."

        games = games.sort_values("start_date")

        game = games.iloc[-1]

        if team_name in str(game["home_team"]).lower():
            team = game["home_team"]
            opponent = game["away_team"]
            team_score = game["home_points"]
            opponent_score = game["away_points"]
        else:
            team = game["away_team"]
            opponent = game["home_team"]
            team_score = game["away_points"]
            opponent_score = game["home_points"]

        if team_score > opponent_score:
            result = "Win"
        elif team_score < opponent_score:
            result = "Loss"
        else:
            result = "Tie"

        return pd.DataFrame([{
            "team": team,
            "opponent": opponent,
            "date": game["start_date"],
            "team_score": team_score,
            "opponent_score": opponent_score,
            "result": result,
        }])