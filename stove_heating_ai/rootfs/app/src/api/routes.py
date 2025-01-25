from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
import os

from core.config import AppConfig
from ml.data import query_influx_data, extract_heating_episodes
from ml.model import train_model, predict, PredictInput, _model_lock
from constants import FluxQueryKeys


logger = logging.getLogger(__name__)

router = APIRouter()

# Mount static files directory
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
router.mount("/static", StaticFiles(directory=static_dir), name="static")


class TrainingStatus:
    def __init__(self):
        self.is_training = False
        self.last_error: str | None = None


training_status = TrainingStatus()


class TrainingResponse(BaseModel):
    status: str
    message: str


class PredictionRequest(BaseModel):
    setpoint_temperature: float
    avg_temperature: float
    living_room_humidity: float
    living_room_temperature: float
    outdoor_temperature: float
    stove_set_power: float
    stove_actual_power: float
    time_since_on: float


class PredictionResponse(BaseModel):
    predicted_minutes: float
    status: str
    message: str


@router.get("/")
async def root(request: Request):
    """Serve the main HTML interface."""
    response = FileResponse(os.path.join(static_dir, "index.html"))
    return response


@router.post("/train", response_model=TrainingResponse)
async def start_training(background_tasks: BackgroundTasks) -> TrainingResponse:
    """Start model training in the background."""
    if training_status.is_training:
        return TrainingResponse(status="error", message="Training already in progress")

    async def train():
        try:
            training_status.is_training = True
            training_status.last_error = None

            config = AppConfig.load()
            df = await query_influx_data(config)
            logger.info("Processed data shape: %s", df.shape)
            logger.debug("Data columns: %s", df.columns.tolist())

            # Use column names from constants
            stove_status_col = FluxQueryKeys.STOVE_STATUS.value
            setpoint_temp_col = FluxQueryKeys.SETPOINT_TEMPERATURE.value
            avg_temp_col = FluxQueryKeys.AVG_TEMPERATURE.value

            logger.debug(
                "Sample of stove_status values: %s",
                df[stove_status_col].value_counts().to_dict(),
            )
            logger.debug(
                "Sample of temperatures - setpoint: %s, avg: %s",
                df[setpoint_temp_col].head(3).tolist(),
                df[avg_temp_col].head(3).tolist(),
            )

            df = extract_heating_episodes(df)
            logger.info("Episodes data shape: %s", df.shape)

            # Drop rows with NaN in Y column (as in original implementation)
            df.dropna(subset=["Y"], inplace=True)
            logger.info("Final data shape after filtering for Y: %s", df.shape)

            if df.empty:
                logger.error("No valid episodes found. This means either:")
                logger.error(
                    "1. No heating episodes were found (stove never turned on)"
                )
                logger.error("2. No episodes reached the comfort temperature")
                logger.error("3. Data is not in expected format")

            with _model_lock:
                train_model(df)

        except Exception as e:
            logger.exception("Training failed")
            training_status.last_error = str(e)
        finally:
            training_status.is_training = False

    background_tasks.add_task(train)
    return TrainingResponse(status="success", message="Training started")


@router.get("/status", response_model=TrainingResponse)
async def get_status() -> TrainingResponse:
    """Get current training status."""
    if training_status.is_training:
        return TrainingResponse(status="running", message="Training in progress")
    elif training_status.last_error:
        return TrainingResponse(
            status="error",
            message=f"Last training failed: {training_status.last_error}",
        )
    return TrainingResponse(status="idle", message="No training in progress")


@router.post("/predict", response_model=PredictionResponse)
async def get_prediction(request: PredictionRequest) -> PredictionResponse:
    """Get a prediction for time to reach comfort temperature."""
    try:
        # Convert request to PredictInput
        input_params = PredictInput(
            setpoint_temperature=request.setpoint_temperature,
            avg_temperature=request.avg_temperature,
            living_room_humidity=request.living_room_humidity,
            living_room_temperature=request.living_room_temperature,
            outdoor_temperature=request.outdoor_temperature,
            stove_set_power=request.stove_set_power,
            stove_actual_power=request.stove_actual_power,
            time_since_on=request.time_since_on,
        )

        predicted_time = predict(input_params)

        return PredictionResponse(
            predicted_minutes=predicted_time,
            status="success",
            message="Prediction successful",
        )
    except Exception as e:
        logger.exception("Prediction failed")
        return PredictionResponse(
            predicted_minutes=0.0,
            status="error",
            message=f"Prediction failed: {str(e)}",
        )


class UpdateSensorResult(BaseModel):
    status: str
    message: str
