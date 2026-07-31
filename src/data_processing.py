import pandas as pd
from nba_api.stats.endpoints import playergamelog


def get_player_game_logs(player_id, season):
    """
    Retrieve one player's regular-season game log from the NBA API.

    Parameters
    ----------
    player_id : int
        The NBA player's unique ID.
    season : str
        The NBA season in YYYY-YY format, such as "2025-26".

    Returns
    -------
    pandas.DataFrame
        The player's games ordered from oldest to newest.
    """
    game_log = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star="Regular Season"
    )

    games = game_log.get_data_frames()[0]

    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

    games = games.sort_values("GAME_DATE").reset_index(drop=True)

    return games

def engineer_features(games):
    """
    Create pre-game features using only statistics from earlier games.

    Parameters
    ----------
    games : pandas.DataFrame
        One player's games ordered from oldest to newest.

    Returns
    -------
    pandas.DataFrame
        The game log with engineered historical features.
    """
    games = games.copy()

    games["IS_HOME"] = games["MATCHUP"].str.contains("vs.").astype(int)

    games["OPPONENT"] = games["MATCHUP"].str[-3:]

    games["REST_DAYS"] = games["GAME_DATE"].diff().dt.days

    # Number of games played before the current game
    games["GAMES_PLAYED"] = range(len(games))

    for stat in ["PTS", "REB", "AST", "MIN", "FGA", "FG3A", "FTA"]:
        previous_games = games[stat].shift(1)

        games[f"{stat}_LAST_GAME"] = previous_games

        games[f"{stat}_LAST_5_AVG"] = (
            previous_games
            .rolling(window=5, min_periods=1)
            .mean()
        )

        games[f"{stat}_LAST_10_AVG"] = (
            previous_games
            .rolling(window=10, min_periods=1)
            .mean()
        )

        games[f"{stat}_SEASON_AVG"] = (
            previous_games
            .expanding()
            .mean()
        )

        games[f"{stat}_STD_LAST_5"] = (
            previous_games
            .rolling(window=5, min_periods=2)
            .std()
        )

        games[f"{stat}_STD_LAST_10"] = (
            previous_games
            .rolling(window=10, min_periods=2)
            .std()
        )

    return games

def build_player_dataset(player_id, season):
    """
    Retrieve and prepare one player's game data.

    Parameters
    ----------
    player_id : int
        The NBA player's unique ID.
    season : str
        The NBA season in YYYY-YY format.

    Returns
    -------
    pandas.DataFrame
        The player's game log with engineered features.
    """
    games = get_player_game_logs(
        player_id=player_id,
        season=season
    )

    games["PLAYER_ID"] = str(player_id)

    games = engineer_features(games)

    return games