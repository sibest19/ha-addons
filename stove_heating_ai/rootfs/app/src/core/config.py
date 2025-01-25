import json
import logging
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Base class for configuration-related errors."""

    pass


class MissingFieldError(ConfigurationError):
    """Raised when required fields are missing in the JSON config."""

    def __init__(self, field: str) -> None:
        super().__init__(f"Missing required field: {field}")
        self.field = field


@dataclass
class AppConfig:
    """Application configuration."""

    influxdb_host: str
    influxdb_port: int
    influxdb_user: str
    influxdb_password: str

    @classmethod
    def load(cls) -> "AppConfig":
        """Load configuration from /data/options.json"""
        try:
            with open("/data/options.json", "r") as file:
                json_data = json.load(file)

                # Validate required fields
                for field_def in fields(cls):
                    if field_def.name not in json_data:
                        raise MissingFieldError(field_def.name)

                # Apply defaults
                defaults: Dict[str, Any] = {"influxdb_port": 8086}
                for key, value in defaults.items():
                    json_data.setdefault(key, value)

                return cls(**json_data)

        except FileNotFoundError:
            raise ConfigurationError("Configuration file not found")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON format: {e}")
