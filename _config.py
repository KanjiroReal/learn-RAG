import os
from enum import Enum
from typing import Dict, Any

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from _logger import logger

class AvailableTools(Enum):
    TRANSLATE = "translate"
    
class ToolStatus(Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"

class ModelType(str, Enum):
    CHAT = "chat"
    VL = "vl"

class ToolConfig(BaseModel):
    name: str
    status: ToolStatus
    function: Any
    description: str | None = None

class ModelConfig(BaseModel):
    name: str
    base_url: str
    api_key: str = "no_key"

class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    chat_private_model_name: str
    chat_private_model_url: str    
    vl_private_model_name: str
    vl_private_model_url: str

class TraceSetting(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str

def load_models_config() -> Dict[ModelType, ModelConfig]:
    settings = ModelSettings() # type: ignore
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

def load_trace_settings():
    """Load langfuse setting into working environment"""
    logger.info("loading langfuse...")
    import nest_asyncio
    import base64
    import logfire
    nest_asyncio.apply()
    
    settings = TraceSetting() # type: ignore
    
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    os.environ["LANGFUSE_HOST"] = settings.langfuse_host
    
    # auth opentelemetry to langfuse
    LANGFUSE_AUTH = base64.b64encode(f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = os.environ.get("LANGFUSE_HOST", "") + "/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic%20{LANGFUSE_AUTH}"
    
    # FIXME: fix log warning
    os.environ["OPENAI_API_KEY"] = "" 
    logfire.configure(service_name='my_agent_service',send_to_logfire=False,)
    logfire.instrument_openai_agents()
    logger.success("Langfuse loaded.")