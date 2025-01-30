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
    An episode represents a complete heating cycle where:
    - Start: When stove_status becomes > 0 (stove turns on)
    - End: When stove_status returns to 0 (stove turns off)
    - Comfort point: When average temperature reaches or exceeds setpoint temperature

    Args:
        df (pd.DataFrame): Input DataFrame containing time series data with required columns:
                          - stove_status: Current state of the stove (0=off, >0=on)
                          - setpoint_temperature: Target temperature
                          - avg_temperature: Current average temperature

    Returns:
        pd.DataFrame: Processed DataFrame containing heating episodes with additional features
                     Returns empty DataFrame if no valid episodes are found
    """
    logger.info("Starting heating episodes extraction process")

    if df.empty:
        logger.warning("Input DataFrame is empty")
        return pd.DataFrame()

    # Create a copy to avoid modifying the original DataFrame
    logger.debug("Creating DataFrame copy and ensuring proper sorting")
    df = df.copy()
    df.sort_index(inplace=True)

    # Get column names from constants
    stove_status_col = FluxQueryKeys.STOVE_STATUS.value
    setpoint_temp_col = FluxQueryKeys.SETPOINT_TEMPERATURE.value
    avg_temp_col = FluxQueryKeys.AVG_TEMPERATURE.value

    # Validate required columns
    required_cols = [stove_status_col, setpoint_temp_col, avg_temp_col]
    logger.debug("Validating required columns: %s", required_cols)
    if not all(col in df.columns for col in required_cols):
        logger.error(
            "DataFrame missing required columns. Available columns: %s", df.columns
        )
        return pd.DataFrame()

    # Convert stove_status to numeric values
    logger.debug("Converting stove status to numeric values")
    df[stove_status_col] = pd.to_numeric(df[stove_status_col], errors="coerce")

    # Detect episode boundaries (when stove turns on)
    logger.debug("Detecting episode start points")
    status_changes = (df[stove_status_col] > 0) & (
        df[stove_status_col].shift(1, fill_value=0) <= 0
    )
    episode_starts = df.index[status_changes]

    logger.info("Found %d potential heating episodes", len(episode_starts))
    if len(episode_starts) == 0:
        logger.warning("No heating episodes detected in the data")
        return pd.DataFrame()

    episodes = []
    total_episodes = 0
    episodes_with_comfort = 0

    # Process each episode
    for start_idx in episode_starts:
        total_episodes += 1
        logger.info("Processing episode %d starting at %s", total_episodes, start_idx)

        # Find episode end point
        episode_slice = df.loc[start_idx:]
        end_mask = episode_slice[stove_status_col] <= 0

        if end_mask.any():
            end_idx = end_mask.idxmax()
            logger.debug(
                "Episode %d ends at %s (stove turned off)", total_episodes, end_idx
            )
        else:
            end_idx = episode_slice.index[-1]
            logger.debug("Episode %d ends at %s (end of data)", total_episodes, end_idx)

        # Extract episode data
        episode_df = df.loc[start_idx:end_idx].copy()
        logger.debug("Episode %d duration: %s", total_episodes, end_idx - start_idx)

        # Calculate minutes since episode start
        episode_df[time_since_on_key] = (
            episode_df.index - episode_df.index[0]
        ).total_seconds() / 60.0

        # Analyze comfort temperature achievement
        comfort_reached_mask = episode_df[avg_temp_col] >= episode_df[setpoint_temp_col]
        comfort_indices = comfort_reached_mask[comfort_reached_mask].index

        if len(comfort_indices) > 0:
            episodes_with_comfort += 1
            comfort_time = comfort_indices[0]

            logger.info(
                "Episode %d reached comfort temperature at %s (%.2f minutes from start)",
                total_episodes,
                comfort_time,
                (comfort_time - start_idx).total_seconds() / 60.0,
            )

            # Calculate time-to-comfort for each point before comfort is reached
            valid_mask = episode_df.index <= comfort_time
            episode_df["Y"] = np.nan
            episode_df.loc[valid_mask, "Y"] = (
                comfort_time - episode_df.index[valid_mask]
            ).total_seconds() / 60.0

            logger.debug(
                "Episode %d temperature progress: Start=%.2f°C, Comfort=%.2f°C, Target=%.2f°C",
                total_episodes,
                episode_df[avg_temp_col].iloc[0],
                episode_df[avg_temp_col][comfort_time],
                episode_df[setpoint_temp_col][comfort_time],
            )

            # Remove the last row if stove is off
            if not episode_df.empty and episode_df.iloc[-1][stove_status_col] == 0:
                logger.debug("Removing final row where stove is off")
                episode_df.drop(episode_df.tail(1).index, inplace=True)

            episodes.append(episode_df)
        else:
            logger.warning(
                "Episode %d did not reach comfort temperature. Max temp=%.2f°C, Target=%.2f°C",
                total_episodes,
                episode_df[avg_temp_col].max(),
                episode_df[setpoint_temp_col].iloc[0],
            )

    # Combine all valid episodes
    if episodes:
        final_df = pd.concat(episodes)
        logger.info(
            "Successfully processed %d/%d episodes that reached comfort temperature",
            episodes_with_comfort,
            total_episodes,
        )
        logger.info("Final dataset shape: %s", final_df.shape)
    else:
        final_df = pd.DataFrame()
        logger.warning(
            "No valid heating episodes found out of %d total episodes", total_episodes
        )

    return final_df
