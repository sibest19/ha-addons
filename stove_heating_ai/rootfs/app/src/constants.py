from enum import Enum


class FluxQueryKeys(Enum):
    TIME = "_time"
    AVG_TEMPERATURE = "avg_temperature"
    LIVING_ROOM_HUMIDITY = "living_room_humidity"
    LIVING_ROOM_TEMPERATURE = "living_room_temperature"
    OUTDOOR_TEMPERATURE = "outdoor_temperature"
    SETPOINT_TEMPERATURE = "setpoint_temperature"
    STOVE_SET_POWER = "stove_set_power"
    STOVE_ACTUAL_POWER = "stove_actual_power"
    STOVE_STATUS = "stove_status"


time_since_on_key = "time_since_on"

# "/data" is a persistent volume in the docker container
model_save_path = "/data/model.keras"
scaler_save_path = "/data/scaler.pkl"


class HomeAssistantSensor(Enum):
    CURRENT_PREDICTION = "sensor.stove_heating_ai_current_prediction"
    LAST_TRAINED = "sensor.stove_heating_ai_last_trained"
    MODEL_TRAINED = "binary_sensor.stove_heating_ai_model_trained"
