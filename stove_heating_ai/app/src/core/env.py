import os


def get_homeassistant_auth_token():
    """Get the Home Assistant API token from the environment."""
    return os.environ["SUPERVISOR_TOKEN"]
