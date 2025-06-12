from enum import Enum
from typing import Dict

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelType(str, Enum):
    CHAT = "chat"
    VL = "vl"
    
class ModelConfig(BaseModel):
    name: str
    base_url: str
    api_key: str = "no_key"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    chat_private_model_name: str
    chat_private_model_url: str    
    vl_private_model_name: str
    vl_private_model_url: str

def load_model_config() -> Dict[ModelType, ModelConfig]:
    settings = Settings() # type: ignore
    return {
        ModelType.CHAT: ModelConfig(
            name=settings.chat_private_model_name,
            base_url=settings.chat_private_model_url
        ),
        ModelType.VL: ModelConfig(
            name=settings.vl_private_model_name,
            base_url=settings.vl_private_model_url
        )
    }