import os
import logging


logger = logging.getLogger(__name__)


def get_homeassistant_auth_token():
    """Get the Home Assistant API token from the environment."""
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        logger.error("SUPERVISOR_TOKEN environment variable not set")
        raise ValueError("Home Assistant authentication token not found")

    if not isinstance(token, str) or len(token) < 32:  # Typical HA tokens are long
        logger.warning("Potentially invalid Home Assistant token format")

    logger.debug("Home Assistant auth token retrieved successfully")
    return token
