import asyncio
import os

from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig

from _config import load_models_config, ModelType

_embedding = None

class AgentManager:
    """Manager for Agent and client of OpenAI Agent SDK"""

    def __init__(self) -> None:
        self.config = load_models_config()
        self.clients = {}
        self._init_clients()
    
    def _init_clients(self):
        """Initialize OpenAI client with different ModelType"""
        for model_type in self.config:
            client_config = self.config[model_type]
            
            self.clients[model_type] = AsyncOpenAI(
                base_url=client_config.base_url,
                api_key=client_config.api_key
            )

    def create_agent(self, name:str, instruction:str, model_type: ModelType, tools: list = []) -> Agent:
        client = self.get_client(model_type)
        model = OpenAIChatCompletionsModel(
            model=self.config[model_type].name,
            openai_client=client
        )
        agent = Agent(
            name=name,
            instructions=instruction,
            model=model,
            tools=tools
        )
        return agent

    async def _run_agent(self, agent: Agent, prompt) -> str:
        result = await Runner.run(agent, prompt, run_config=RunConfig(tracing_disabled=True))
        return result.final_output
    
    def run_agent(self, agent: Agent, prompt) -> str:
        return asyncio.run(self._run_agent(agent, prompt))
    
    def get_client(self, client_type: ModelType) -> AsyncOpenAI:
        return self.clients[client_type]

def get_embedding():
    global _embedding
    if _embedding is None:
        _embedding = SentenceTransformer("huyydangg/DEk21_hcmute_embedding")
    return _embedding


agent_manager = AgentManager()