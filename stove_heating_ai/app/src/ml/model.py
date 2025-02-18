from typing import Optional, Tuple, Any
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pydantic import BaseModel, Field
import keras
from threading import Lock
import tensorflow as tf

from constants import (
    model_save_path,
    FluxQueryKeys,
    time_since_on_key,
    scaler_save_path,
    training_data_path,
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

# Create a global lock
_model_lock = Lock()


def create_model(input_shape: int) -> keras.Sequential:
    with tf.device("/CPU:0"):
        return keras.Sequential(
            [
                keras.layers.InputLayer(shape=(input_shape,)),
                keras.layers.Dense(64, activation="elu"),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(1),
            ]
        )


def create_model_variation(input_shape: int, architecture: str) -> keras.Sequential:
    """Create and return different neural network model architectures."""

    with tf.device("/CPU:0"):
        if architecture == "simple":
            return keras.Sequential(
                [
                    keras.layers.InputLayer(shape=(input_shape,)),
                    keras.layers.Dense(32, activation="relu"),
                    keras.layers.Dense(32, activation="relu"),
                    keras.layers.Dense(1),
                ]
            )
        elif architecture == "deep":
            return keras.Sequential(
                [
                    keras.layers.InputLayer(shape=(input_shape,)),
                    keras.layers.Dense(64, activation="relu"),
                    keras.layers.Dense(128, activation="relu"),
                    keras.layers.Dense(128, activation="relu"),
                    keras.layers.Dense(64, activation="relu"),
                    keras.layers.Dense(1),
                ]
            )
        elif architecture == "elu":
            return keras.Sequential(
                [
                    keras.layers.InputLayer(shape=(input_shape,)),
                    keras.layers.Dense(64, activation="elu"),
                    keras.layers.Dense(128, activation="elu"),
                    keras.layers.Dense(64, activation="elu"),
                    keras.layers.Dense(1),
                ]
            )
        elif architecture == "wide":
            return keras.Sequential(
                [
                    keras.layers.InputLayer(shape=(input_shape,)),
                    keras.layers.Dense(256, activation="relu"),
                    keras.layers.Dense(256, activation="relu"),
                    keras.layers.Dense(1),
                ]
            )
        elif architecture == "residual":
            # Residual connections can help with gradient flow
            inputs = keras.layers.Input(shape=(input_shape,))
            x = keras.layers.Dense(64, activation="relu")(inputs)
            residual = x
            x = keras.layers.Dense(64, activation="relu")(x)
            x = keras.layers.Add()([x, residual])
            x = keras.layers.Dense(32, activation="relu")(x)
            outputs = keras.layers.Dense(1)(x)
            return keras.Model(inputs=inputs, outputs=outputs)
        elif architecture == "regularized":
            # L1L2 regularization to prevent overfitting
            return keras.Sequential(
                [
                    keras.layers.InputLayer(shape=(input_shape,)),
                    keras.layers.Dense(
                        64,
                        activation="relu",
                        kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    ),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(
                        128,
                        activation="relu",
                        kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    ),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(1),
                ]
            )
        elif architecture == "selu":
            # Self-normalizing neural network
            return keras.Sequential(
                [
                    keras.layers.InputLayer(shape=(input_shape,)),
                    keras.layers.Dense(
                        64, activation="selu", kernel_initializer="lecun_normal"
                    ),
                    keras.layers.Dense(
                        128, activation="selu", kernel_initializer="lecun_normal"
                    ),
                    keras.layers.Dense(
                        64, activation="selu", kernel_initializer="lecun_normal"
                    ),
                    keras.layers.Dense(1),
                ]
            )
        elif architecture == "deep_dropout":
            # Deeper network with aggressive dropout
            return keras.Sequential(
                [
                    keras.layers.InputLayer(shape=(input_shape,)),
                    keras.layers.Dense(128, activation="relu"),
                    keras.layers.Dropout(0.3),
                    keras.layers.Dense(256, activation="relu"),
                    keras.layers.Dropout(0.3),
                    keras.layers.Dense(256, activation="relu"),
                    keras.layers.Dropout(0.3),
                    keras.layers.Dense(128, activation="relu"),
                    keras.layers.Dropout(0.3),
                    keras.layers.Dense(1),
                ]
            )
        elif architecture == "leaky":
            # Using LeakyReLU for potential better gradient flow
            return keras.Sequential(
                [
                    keras.layers.InputLayer(shape=(input_shape,)),
                    keras.layers.Dense(64),
                    keras.layers.LeakyReLU(alpha=0.1),
                    keras.layers.Dense(128),
                    keras.layers.LeakyReLU(alpha=0.1),
                    keras.layers.Dense(64),
                    keras.layers.LeakyReLU(alpha=0.1),
                    keras.layers.Dense(1),
                ]
            )
        else:  # default to original
            return create_model(input_shape)


def train_model(df: pd.DataFrame) -> Tuple[Optional[keras.Sequential], Any]:
    """Train the model on provided data."""

    with _model_lock:
        if df.empty:
            logger.warning("No data available for training")
            return None, None

        feature_columns = FEATURE_COLUMNS

        missing_columns = set(FEATURE_COLUMNS) - set(df.columns)
        if missing_columns:
            logger.error("Missing required columns: %s", missing_columns)
            return None, None

        X = df[feature_columns]
        y = df["Y"]

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        from sklearn.model_selection import KFold

        kfold = KFold(n_splits=5, shuffle=True, random_state=4765)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        architectures = [
            "original",
            "simple",
            "deep",
            "elu",
            "wide",
            "residual",
            "regularized",
            "selu",
            "deep_dropout",
            "leaky",
        ]
        architecture_scores = {}
        best_model = None
        best_score = float("inf")
        best_architecture = None

        for architecture in architectures:
            logger.info(f"Training architecture: {architecture}")
            val_scores = []

            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled)):
                X_train_fold, X_val_fold = (
                    X_train_scaled[train_idx],
                    X_train_scaled[val_idx],
                )
                y_train_fold, y_val_fold = (
                    y_train.iloc[train_idx],
                    y_train.iloc[val_idx],
                )

                model = create_model_variation(X_train_scaled.shape[1], architecture)
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
                    X_train_fold,
                    y_train_fold,
                    validation_data=(X_val_fold, y_val_fold),
                    epochs=100,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=0,  # type: ignore
                )

                val_loss = model.evaluate(X_val_fold, y_val_fold, verbose=0)[0]
                val_scores.append(val_loss)
                logger.info(f"{architecture} - Fold {fold + 1}: MSE = {val_loss:.4f}")

            avg_score = sum(val_scores) / len(val_scores)
            architecture_scores[architecture] = avg_score
            logger.info(f"{architecture} - Average MSE: {avg_score:.4f}")

            if avg_score < best_score:
                best_score = avg_score
                best_model = model
                best_architecture = architecture

        logger.info("Architecture comparison:")
        for arch, score in sorted(architecture_scores.items(), key=lambda x: x[1]):
            logger.info(f"{arch:10} - MSE: {score:.4f}")
        logger.info(f"Best architecture: {best_architecture} (MSE: {best_score:.4f})")

        if best_model is None:
            logger.error("Model training failed")
            return None, None

        # Evaluate best model on test set
        test_loss, test_mae = best_model.evaluate(X_test_scaled, y_test, verbose=0)  # type: ignore
        logger.info(
            "Model evaluation on test set - MAE: %.4f, MSE: %.4f", test_mae, test_loss
        )

        # Save both model and scaler
        best_model.save(model_save_path)
        joblib.dump(scaler, scaler_save_path)

        global _model, _scaler

        _model = best_model
        _scaler = scaler

        logger.info(
            "Model saved to %s and scaler to %s", model_save_path, scaler_save_path
        )

        # Save training data visualization
        try:
            styled_df = df.style.background_gradient(subset=["Y", "avg_temperature"])
            html = styled_df.to_html(escape=True)  # Sanitize HTML content
            with open(training_data_path, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info("Training data visualization saved to %s", training_data_path)
        except Exception as e:
            logger.error("Failed to save training data visualization: %s", str(e))

        return best_model, history


class PredictInput(BaseModel):
    """Data Transfer Object for prediction input parameters."""

    setpoint_temperature: float = Field(..., gt=-50, lt=100)
    avg_temperature: float = Field(..., gt=-50, lt=100)
    living_room_humidity: float = Field(..., ge=0, le=100)
    living_room_temperature: float = Field(..., gt=-50, lt=100)
    outdoor_temperature: float = Field(..., gt=-100, lt=100)
    stove_set_power: float = Field(..., ge=0, le=5)
    stove_actual_power: float = Field(..., ge=0, le=5)
    time_since_on: float = Field(..., ge=0)


def predict(input_params: PredictInput) -> float:
    """Make a prediction using the saved model.

    Args:
        input_params: PredictInput object containing all required parameters

    Returns:
        Predicted time to reach comfort temperature in minutes
    """
    global _model, _scaler

    try:
        with _model_lock:
            if _model is None or _scaler is None:
                # Ensure model loading happens on CPU
                with tf.device("/CPU:0"):
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
        prediction = _model.predict(features_scaled, verbose=0)  # type: ignore
        return float(prediction[0][0])

    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        raise
