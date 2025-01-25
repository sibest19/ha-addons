import logging
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
    Extracts heating episodes from the input DataFrame using vectorized operations.
    Each episode starts when stove_status becomes > 0 and ends when it returns to 0.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.sort_index(inplace=True)

    stove_status_col = FluxQueryKeys.STOVE_STATUS.value
    setpoint_temp_col = FluxQueryKeys.SETPOINT_TEMPERATURE.value
    avg_temp_col = FluxQueryKeys.AVG_TEMPERATURE.value

    # Verify required columns exist
    required_cols = [stove_status_col, setpoint_temp_col, avg_temp_col]
    if not all(col in df.columns for col in required_cols):
        logger.error("Missing required columns")
        return pd.DataFrame()

    logger.debug("Starting episode extraction with columns:")
    logger.debug("- Stove status: %s", stove_status_col)
    logger.debug("- Setpoint temp: %s", setpoint_temp_col)
    logger.debug("- Average temp: %s", avg_temp_col)

    # Convert stove_status to numeric if needed
    df[stove_status_col] = pd.to_numeric(df[stove_status_col], errors="coerce")

    # Detect episode boundaries
    status_changes = (df[stove_status_col] > 0) & (
        df[stove_status_col].shift(1, fill_value=0) <= 0
    )
    episode_starts = df.index[status_changes]

    if len(episode_starts) == 0:
        logger.info("No heating episodes found")
        return pd.DataFrame()

    episodes = []
    total_episodes = 0
    episodes_with_comfort = 0

    for start_idx in episode_starts:
        # Find episode end (when stove turns off or data ends)
        episode_slice = df.loc[start_idx:]
        end_mask = episode_slice[stove_status_col] <= 0

        if end_mask.any():
            end_idx = end_mask.idxmax()
        else:
            end_idx = episode_slice.index[-1]

        episode_df = df.loc[start_idx:end_idx].copy()
        total_episodes += 1

        # Calculate time since episode start (vectorized)
        episode_df[time_since_on_key] = (
            episode_df.index - episode_df.index[0]
        ).total_seconds() / 60.0

        # Check if comfort temperature was reached
        comfort_reached_mask = episode_df[avg_temp_col] >= episode_df[setpoint_temp_col]

        # Find the first row in which comfort is reached
        comfort_indices = comfort_reached_mask[comfort_reached_mask].index
        if len(comfort_indices) > 0:
            episodes_with_comfort += 1
            comfort_time = comfort_indices[0]

            logger.debug(
                "Episode %d reached comfort at %s",
                total_episodes,
                comfort_time,
            )

            # Calculate Y values vectorized (only for times before comfort)
            valid_mask = episode_df.index <= comfort_time
            episode_df["Y"] = np.nan
            episode_df.loc[valid_mask, "Y"] = (
                comfort_time - episode_df.index[valid_mask]
            ).total_seconds() / 60.0

            # Drop last row if the stove is off
            if not episode_df.empty and episode_df.iloc[-1][stove_status_col] == 0:
                episode_df.drop(episode_df.tail(1).index, inplace=True)

            episodes.append(episode_df)
        else:
            logger.debug("Episode %d did not reach comfort temperature", total_episodes)

    if episodes:
        final_df = pd.concat(episodes)
        logger.info(
            "Found %d total episodes, %d reached comfort temperature",
            total_episodes,
            episodes_with_comfort,
        )
        logger.debug("Combined episodes shape: %s", final_df.shape)
    else:
        final_df = pd.DataFrame()
        logger.info(
            "Found %d total episodes, but none reached comfort temperature",
            total_episodes,
        )

    return final_df
