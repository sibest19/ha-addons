from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
import os
from concurrent.futures import ProcessPoolExecutor
import asyncio

from core.config import AppConfig
from ml.data import query_influx_data, extract_heating_episodes
from ml.model import train_model, predict, PredictInput
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


def init_worker():
    """Initialize TensorFlow configuration for worker processes."""
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")  # Disable GPU
    physical_devices = tf.config.list_physical_devices("CPU")
    if physical_devices:
        tf.config.set_visible_devices(physical_devices, "CPU")
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)


# Update the process pool creation
def create_process_pool():
    try:
        return ProcessPoolExecutor(
            max_workers=1,  # Limit to single worker for TensorFlow stability
            initializer=init_worker,
        )
    except Exception as e:
        logger.error("Failed to create process pool: %s", str(e))
        # Fallback to synchronous execution
        init_worker()
        return None


# Replace the global process_pool with:
process_pool = create_process_pool()


@router.on_event("shutdown")
async def shutdown_event():
    """Ensure proper cleanup of process pool on shutdown."""
    try:
        if process_pool:
            process_pool.shutdown(wait=True)
    except Exception as e:
        logger.error("Error during process pool shutdown: %s", str(e))


@router.get("/")
async def root(request: Request):
    """Serve the main HTML interface."""
    html_path = os.path.join(static_dir, "index.html")
    if not os.path.exists(html_path):
        logger.error("HTML file not found at %s", html_path)
        raise HTTPException(status_code=500, detail="Interface file not found")
    response = FileResponse(html_path)
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
            # Use shorter timeout for data query
            df = await asyncio.wait_for(query_influx_data(config), timeout=300)

            if df.empty:
                raise ValueError("No data available for training")

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
            else:
                # Run training in a separate process with explicit loop
                loop = asyncio.get_event_loop()
                # Increase timeout to 1 hour for larger datasets
                await asyncio.wait_for(
                    loop.run_in_executor(process_pool, train_model, df), timeout=3600
                )

        except asyncio.TimeoutError:
            logger.error("Training timed out")
            training_status.last_error = "Training timed out"
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
