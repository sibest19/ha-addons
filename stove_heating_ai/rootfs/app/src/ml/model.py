from typing import Optional, Tuple, Any, List
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import keras

from constants import (
    model_save_path,
    FluxQueryKeys,
    time_since_on_key,
    scaler_save_path,
)

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    FluxQueryKeys.SETPOINT_TEMPERATURE.value,
    FluxQueryKeys.AVG_TEMPERATURE.value,
    FluxQueryKeys.LIVING_ROOM_HUMIDITY.value,
    FluxQueryKeys.LIVING_ROOM_TEMPERATURE.value,
    FluxQueryKeys.OUTDOOR_TEMPERATURE.value,
    FluxQueryKeys.STOVE_SET_POWER.value,
    FluxQueryKeys.STOVE_ACTUAL_POWER.value,
    time_since_on_key,
]

_model = None
_scaler = None


def create_model(input_shape: int) -> keras.Sequential:
    """Create and return the neural network model."""
    return keras.Sequential(
        [
            keras.layers.InputLayer(shape=(input_shape,)),
            keras.layers.Dense(64, activation="elu"),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1),
        ]
    )


def train_model(df: pd.DataFrame) -> Tuple[Optional[keras.Sequential], Any]:
    """Train the model on provided data."""
    if df.empty:
        logger.warning("No data available for training")
        return None, None

    feature_columns = FEATURE_COLUMNS

    X = df[feature_columns]
    y = df["Y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=4765
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = create_model(X_train_scaled.shape[1])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=0.0001
        ),
    ]

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose="auto",
    )

    mse, mae = model.evaluate(X_test_scaled, y_test, verbose="auto")
    logger.info("Model evaluation on test set - MAE: %.4f, MSE: %.4f", mae, mse)

    # Save both model and scaler
    model.save(model_save_path)
    joblib.dump(scaler, scaler_save_path)

    logger.info("Model saved to %s and scaler to %s", model_save_path, scaler_save_path)

    return model, history


class PredictInput:
    """Data Transfer Object for prediction input parameters."""

    def __init__(
        self,
        setpoint_temperature: float,
        avg_temperature: float,
        living_room_humidity: float,
        living_room_temperature: float,
        outdoor_temperature: float,
        stove_set_power: float,
        stove_actual_power: float,
        time_since_on: float,
    ):
        self.setpoint_temperature = setpoint_temperature
        self.avg_temperature = avg_temperature
        self.living_room_humidity = living_room_humidity
        self.living_room_temperature = living_room_temperature
        self.outdoor_temperature = outdoor_temperature
        self.stove_set_power = stove_set_power
        self.stove_actual_power = stove_actual_power
        self.time_since_on = time_since_on


def predict(input_params: PredictInput) -> float:
    """Make a prediction using the saved model.

    Args:
        input_params: PredictInput object containing all required parameters

    Returns:
        Predicted time to reach comfort temperature in minutes
    """
    global _model, _scaler

    try:
        if _model is None or _scaler is None:
            _model = keras.models.load_model(model_save_path)
            _scaler = joblib.load(scaler_save_path)

        if not isinstance(_model, keras.Sequential):
            logger.error("Invalid model")
            raise ValueError("Invalid model type")

        if not isinstance(_scaler, StandardScaler):
            logger.error("Invalid scaler")
            raise ValueError("Invalid scaler type")

        # Convert input parameters to DataFrame with feature names
        features = pd.DataFrame(
            [
                [
                    input_params.setpoint_temperature,
                    input_params.avg_temperature,
                    input_params.living_room_humidity,
                    input_params.living_room_temperature,
                    input_params.outdoor_temperature,
                    input_params.stove_set_power,
                    input_params.stove_actual_power,
                    input_params.time_since_on,
                ]
            ],
            columns=FEATURE_COLUMNS,
        )

        # Scale features using the saved scaler
        features_scaled = _scaler.transform(features)

        # Make prediction
        prediction = _model.predict(features_scaled, verbose="auto")
        return float(prediction[0][0])

    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        raise
