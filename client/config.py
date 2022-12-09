from functools import lru_cache
from pydantic import BaseSettings
from typing import Union
import enum

class ProductionMode(enum.Enum):
    PRODUCTION = "prod"
    DEVELOPMENT = "dev"


class Settings(BaseSettings):
    mode : ProductionMode = ProductionMode.DEVELOPMENT

    server_ip: str
    server_port: int
    frame_send_port: int
    car_receive_port: int

    width: int
    height: int
    framerate: int
    jpeg_quality: int

    client_os: str = ""

    car_id: str

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()
