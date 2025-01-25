import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from constants import FluxQueryKeys, time_since_on_key
from ml.data import extract_heating_episodes


@pytest.fixture
def sample_data():
    # Create sample data for 2 hours with 1-minute intervals
    dates = pd.date_range(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 1, 1, 59),
        freq="min"
    )

    # Initialize all values
    data = pd.DataFrame({
        'stove_status': np.zeros(len(dates)),  # Initially all off
        'setpoint_temperature': np.full(len(dates), 21.0),  # Target temp
        'avg_temperature': np.full(len(dates), 18.0),  # Starting temp
    }, index=dates)

    # Create a heating episode from 00:10 to 00:30
    episode1_mask = (data.index >= '2024-01-01 00:10:00') & (data.index < '2024-01-01 00:30:00')
    data.loc[episode1_mask, 'stove_status'] = 1

    # Temperature gradually increases during this episode until comfort is reached at 00:25
    temps = np.linspace(18.0, 21.5, 20)  # 20 minutes from 18°C to 21.5°C
    data.loc[episode1_mask, 'avg_temperature'] = temps

    # Create another episode from 01:00 to 01:15 that doesn't reach comfort
    episode2_mask = (data.index >= '2024-01-01 01:00:00') & (data.index < '2024-01-01 01:15:00')
    data.loc[episode2_mask, 'stove_status'] = 1
    temps2 = np.linspace(18.0, 20.0, 15)  # 15 minutes from 18°C to 20°C (doesn't reach setpoint)
    data.loc[episode2_mask, 'avg_temperature'] = temps2

    return data


def test_extract_heating_episodes_basic(sample_data):
    result = extract_heating_episodes(sample_data)

    # Should only contain the first episode that reached comfort
    assert len(result) == 20  # First episode length
    assert "Y" in result.columns
    assert time_since_on_key in result.columns

    # Check Y values are decreasing
    y_values = result["Y"].dropna()
    assert len(y_values) > 0
    # Verify Y values are strictly decreasing
    assert all(y_values.iloc[i] > y_values.iloc[i+1] for i in range(len(y_values)-1))


def test_extract_heating_episodes_no_comfort(sample_data):
    # Modify sample data so no episode reaches comfort
    sample_data['avg_temperature'] = 18.0
    result = extract_heating_episodes(sample_data)
    assert result.empty


def test_extract_heating_episodes_empty():
    df = pd.DataFrame()
    result = extract_heating_episodes(df)
    assert len(result) == 0


def test_extract_heating_episodes_multiple_episodes(sample_data):
    result = extract_heating_episodes(sample_data)

    # Check that only episodes reaching comfort are included
    unique_episodes = result.index.to_series().diff().dt.total_seconds().abs() > 60
    n_episodes = unique_episodes.sum() + 1 if len(result) > 0 else 0
    assert n_episodes == 1  # Only first episode should be included


def test_extract_heating_episodes_time_values(sample_data):
    result = extract_heating_episodes(sample_data)

    # Get first episode
    first_episode = result[result.index < "2024-01-01 00:30:00"]

    # Check time_since_on starts at 0
    assert first_episode[time_since_on_key].iloc[0] == 0

    # Check time_since_on ends at approximately 19 (20 minutes - 1)
    assert abs(first_episode[time_since_on_key].iloc[-1] - 19) < 0.1
