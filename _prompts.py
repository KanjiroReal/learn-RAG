import os
from enum import Enum

from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

from _logger import logger


class PROMPT(Enum):
    """All Prompts filename for agent. Each will store a filename of its prompt
    
    Naming convention:
        [service]_[Agent name]
    
    Example:
        PARSER_SUMMARIZER
    """
    PARSER_SUMMARIZER = "PARSER_SUMMARIZER"
    RAG_RETRIEVAL_ASSISTANT = "RAG_RETRIEVAL_ASSISTANT"
    OCR_CONTEXT = "OCR_CONTEXT"


class PromptManager:
    def __init__(self) -> None:
        self.prompts = {}
        self._load_prompt()
        
    def _load_prompt(self):
        directory = "prompts"
        for item_name in os.listdir(directory):
            item_path = os.path.join(directory, item_name)
            if os.path.isfile(item_path):
                if item_path.endswith(".txt"):
                    with open(item_path, "r", encoding='utf-8') as f:
                        key = item_name.split(".")[0].upper()
                        self.prompts[key] = f.read()
    
    def get_prompt(self, prompt_name: PROMPT, agentic_wraper:bool = False) -> str:
        try:
            prompt = self.prompts[prompt_name.value]
            if agentic_wraper:
                prompt = prompt_with_handoff_instructions(prompt=prompt)
            return prompt
        except Exception as e:
            logger.critical(f"Lỗi khi truy xuất prompt: {e}")
            return ""

prompt_manager = PromptManager()