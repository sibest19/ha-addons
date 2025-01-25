import logging
from typing import Optional, List
import os

import numpy as np
import pandas as pd
from influxdb_client.client.influxdb_client import InfluxDBClient
import asyncio

from core.config import AppConfig
from constants import FluxQueryKeys, time_since_on_key

logger = logging.getLogger(__name__)


async def query_influx_data(configuration: AppConfig | None) -> pd.DataFrame:
    """
    Queries InfluxDB for relevant time series data and returns a Pandas DataFrame.
    Raises a ValueError if global configuration is not loaded.

    Returns:
        A Pandas DataFrame containing the time series data.
    """
    if configuration is None:
        raise ValueError("Configuration not loaded.")
    logger.info(
        "Establishing connection to InfluxDB at %s", configuration.influxdb_host
    )

    client = InfluxDBClient(
        url=f"http://{configuration.influxdb_host}:{configuration.influxdb_port}",
        token=f"{configuration.influxdb_user}:{configuration.influxdb_password}",
        auth_basic=True,
        debug=False,
    )

    try:
        flux_query_path = os.path.join(os.path.dirname(__file__), "query.flux")
        with open(flux_query_path, "r") as file:
            flux_query = file.read()

        logger.debug("Flux query:\n%s", flux_query)

        query_api = client.query_api()
        # Use asyncio.to_thread for blocking operations
        raw_df: pd.DataFrame = await asyncio.to_thread(
            query_api.query_data_frame, query=flux_query
        )  # type: ignore
        if "_time" in raw_df.columns:
            raw_df["_time"] = pd.to_datetime(raw_df["_time"])
            raw_df.sort_values(by="_time", inplace=True)
            raw_df.set_index("_time", inplace=True)

        logger.debug("Raw DataFrame shape: %s", raw_df.shape)

        # Process data exactly as in the original implementation
        logger.debug("Resampling data to 1-minute intervals with forward fill.")
        df_resampled = (
            raw_df.drop(columns=["result", "Unnamed: 0", "table"], errors="ignore")
            .resample("1min")
            .mean()
            .ffill()
        )
        logger.debug("Resampled DataFrame shape: %s", df_resampled.shape)

        df_resampled.dropna(inplace=True)
        logger.debug("Shape after dropping NaNs: %s", df_resampled.shape)

        return df_resampled
    finally:
        await asyncio.to_thread(client.close)


def extract_heating_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts heating episodes from the input DataFrame.
    Each episode starts when stove_status becomes > 0 (previously <= 0),
    and ends when stove_status goes back to 0 or data ends.

    If, during the episode, the comfort temperature is reached, a new 'Y' column is added:
      Y = the time (minutes) remaining from the current row to the time comfort is reached.
      If the row time is past the comfort time, Y=NaN.

    Adds a 'time_since_on' column to track minutes elapsed from the start of the episode.
    Discards episodes that never reach comfort temperature.

    Args:
        df: A pandas DataFrame containing time series data with stove status,
            setpoint temperature, average temperature, etc.

    Returns:
        A new DataFrame containing only valid episodes (those that reach comfort).
        If no valid episodes exist, returns an empty DataFrame.
    """
    df = df.copy()  # do not mutate the original DataFrame
    df.sort_index(inplace=True)

    episodes: List[pd.DataFrame] = []
    i = 0
    n = len(df)
    total_episodes_found = 0
    episodes_reaching_comfort = 0

    stove_status_col = FluxQueryKeys.STOVE_STATUS.value
    setpoint_temp_col = FluxQueryKeys.SETPOINT_TEMPERATURE.value
    avg_temp_col = FluxQueryKeys.AVG_TEMPERATURE.value

    logger.debug("Starting episode extraction with columns:")
    logger.debug("- Stove status: %s", stove_status_col)
    logger.debug("- Setpoint temp: %s", setpoint_temp_col)
    logger.debug("- Average temp: %s", avg_temp_col)

    while i < n:
        row = df.iloc[i]
        stove_status = row[stove_status_col]

        if stove_status > 0:
            start_idx = i
            j = i
            while j < n:
                stove_status_j = df.iloc[j][stove_status_col]
                if stove_status_j <= 0:
                    break
                j += 1

            total_episodes_found += 1
            logger.debug(
                "Found episode %d from index %d to %d",
                total_episodes_found,
                start_idx,
                j,
            )

            episode_df = df.iloc[start_idx:j].copy()
            episode_df[time_since_on_key] = 0.0

            start_time = episode_df.index[0]
            for k in range(len(episode_df)):
                row_time = episode_df.index[k]
                delta_since_start = (row_time - start_time).total_seconds() / 60.0
                episode_df.iat[k, episode_df.columns.get_loc(time_since_on_key)] = (
                    delta_since_start
                )

            comfort_reached_mask = (
                episode_df[avg_temp_col] >= episode_df[setpoint_temp_col]
            )
            if comfort_reached_mask.any():
                episodes_reaching_comfort += 1
                first_reach_idx = comfort_reached_mask.idxmax()
                t_comfort_reached = episode_df.loc[first_reach_idx].name
                logger.debug(
                    "Episode %d reached comfort at %s",
                    total_episodes_found,
                    t_comfort_reached,
                )

                episode_df["Y"] = np.nan
                for k in range(len(episode_df)):
                    row_time = episode_df.index[k]
                    delta = (t_comfort_reached - row_time).total_seconds() / 60.0
                    if delta < 0:
                        episode_df.iat[k, episode_df.columns.get_loc("Y")] = np.nan
                    else:
                        episode_df.iat[k, episode_df.columns.get_loc("Y")] = delta

                episodes.append(episode_df)
            else:
                logger.debug(
                    "Episode %d did not reach comfort temperature", total_episodes_found
                )

            i = j
        else:
            i += 1

    if episodes:
        final_df = pd.concat(episodes)
        logger.info(
            "Found %d total episodes, %d reached comfort temperature",
            total_episodes_found,
            episodes_reaching_comfort,
        )
        logger.debug("Combined episodes shape: %s", final_df.shape)
    else:
        final_df = pd.DataFrame()
        logger.info(
            "Found %d total episodes, but none reached comfort temperature",
            total_episodes_found,
        )

    return final_df
