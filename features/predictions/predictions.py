from kickbase_api.league import get_league_players_on_market
from kickbase_api.user import get_players_in_squad
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

def live_data_predictions(today_df, model, features):
    """Make live data predictions for today_df using the trained model"""

    # Set features and copy df
    today_df_features = today_df[features]
    today_df_results = today_df.copy()

    # Predict mv_target
    today_df_results["predicted_mv_target"] = np.round(model.predict(today_df_features), 2)

    # Sort by predicted_mv_target descending
    today_df_results = today_df_results.sort_values("predicted_mv_target", ascending=False)

    # Filter date to today or yesterday if before 22:15, because mv is updated around 22:15
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    cutoff_time = now.replace(hour=22, minute=15, second=0, microsecond=0)
    date = (now - timedelta(days=1)) if now <= cutoff_time else now
    date = date.date()

    # Drop rows where NaN mv
    today_df_results = today_df_results.dropna(subset=["mv"])

    # Keep only relevant columns
    today_df_results = today_df_results[["player_id", "first_name", "last_name", "position", "team_name", "date", "mv_change_1d", "mv_trend_1d", "mv", "predicted_mv_target"]]

    return today_df_results


def estimate_start_11_from_minutes(player_df, last_games=5):
    """Heuristic only: share of the last completed matchdays with >=60 minutes.

    This is not the official Kickbase starting-XI probability and cannot
    reliably distinguish starters from substitute appearances.
    """
    games = player_df[["player_id", "md", "mp"]].copy()
    games["md"] = pd.to_datetime(games["md"], errors="coerce")
    games["mp"] = pd.to_numeric(games["mp"], errors="coerce")
    today = pd.Timestamp(datetime.now(ZoneInfo("Europe/Berlin")).date())
    games = games[(games["md"] < today) & games["md"].notna() & games["mp"].notna()]
    games = games.sort_values(["player_id", "md"]).drop_duplicates(
        subset=["player_id", "md"], keep="last"
    )
    games = games.groupby("player_id", group_keys=False).tail(last_games)
    result = games.groupby("player_id")["mp"].agg(
        observed_games="count",
        sixty_min_games=lambda minutes: (minutes >= 60).sum(),
    ).reset_index()
    result["start_11_est_pct"] = (
        100 * result["sixty_min_games"] / result["observed_games"]
    ).round(0)
    return result[["player_id", "start_11_est_pct", "observed_games"]]


def join_current_squad(token, league_id, today_df_results, start_estimates=None):
    squad_players = get_players_in_squad(token, league_id)

    squad_df = pd.DataFrame(squad_players["it"])

    # Join squad_df ("i") with today_df ("player_id")
    squad_df = (
        pd.merge(today_df_results, squad_df, left_on="player_id", right_on="i")
        .drop(columns=["i"])
    )

    # Rename prob to s_11_prob for better understanding
    if "prob" not in squad_df.columns:
        squad_df["prob"] = np.nan  # Placeholder for non-pro users
    squad_df = squad_df.rename(columns={"prob": "s_11_prob"})

    # Rename mv_change_1d to mv_change_yesterday for better understanding
    squad_df = squad_df.rename(columns={"mv_change_1d": "mv_change_yesterday"})

    # Rename "mv_x" to "mv" for better understanding
    squad_df = squad_df.rename(columns={"mv_x": "mv"})

    if start_estimates is not None:
        squad_df = squad_df.merge(start_estimates, on="player_id", how="left")
    else:
        squad_df["start_11_est_pct"] = np.nan
        squad_df["observed_games"] = np.nan

    # Keep official and heuristic estimates separate.
    squad_df = squad_df[["last_name", "team_name", "mv", "mv_change_yesterday",
                         "predicted_mv_target", "s_11_prob",
                         "start_11_est_pct", "observed_games"]]

    return squad_df 


# TODO Add fail-safe check before player expires if the prob (starting 11) is still high, so no injuries or anything. if it dropped. dont bid / reccommend
def join_current_market(token, league_id, today_df_results, start_estimates=None):
    """Join the live predictions with the current market data to get bid recommendations"""

    players_on_market = get_league_players_on_market(token, league_id)

    # players_on_market to DataFrame
    market_df = pd.DataFrame(players_on_market)

    # Join market_df ("id") with today_df ("player_id")
    bid_df = (
        pd.merge(today_df_results, market_df, left_on="player_id", right_on="id")
        .drop(columns=["id"])
    )

    # exp contains seconds until expiration
    bid_df["hours_to_exp"] = np.round((bid_df["exp"] / 3600), 2)

    # check if current sysdate + hours_to_exp is after the next 22:00
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    next_22 = now.replace(hour=22, minute=0, second=0, microsecond=0)
    diff = np.round((next_22 - now).total_seconds() / 3600, 2)

    # If hours_to_exp < diff then it expires today
    bid_df["expiring_today"] = bid_df["hours_to_exp"] < diff

    # Drop rows where predicted_mv_target is less than 5000
    bid_df = bid_df[bid_df["predicted_mv_target"] > 5000]

    # Sort by predicted_mv_target descending
    bid_df = bid_df.sort_values("predicted_mv_target", ascending=False)

    # Rename prob to s_11_prob for better understanding
    if "prob" not in bid_df.columns:
        bid_df["prob"] = np.nan  # Placeholder for non-pro users
    bid_df = bid_df.rename(columns={"prob": "s_11_prob"})

    # Rename mv_change_1d to mv_change_yesterday for better understanding
    bid_df = bid_df.rename(columns={"mv_change_1d": "mv_change_yesterday"})

    if start_estimates is not None:
        bid_df = bid_df.merge(start_estimates, on="player_id", how="left")
    else:
        bid_df["start_11_est_pct"] = np.nan
        bid_df["observed_games"] = np.nan

    # Keep official and heuristic estimates separate.
    bid_df = bid_df[["last_name", "team_name", "mv", "mv_change_yesterday",
                     "predicted_mv_target", "s_11_prob", "start_11_est_pct",
                     "observed_games", "hours_to_exp", "expiring_today"]]

    return bid_df