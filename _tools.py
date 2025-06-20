import re
import base64
from typing import List
from io import BytesIO

import pytesseract
from googletrans import Translator
from agents import function_tool, Tool
from PIL import Image

from _logger import logger
from _config import ToolStatus, AvailableTools, ToolConfig


class ToolsManager:
    def __init__(self) -> None:
        
        # declair tool here
        self.tools_cfg = {
            AvailableTools.TRANSLATE : ToolConfig(
                name="translate",
                status=ToolStatus.ENABLED,
                function=_translate,
            )
        }

    def get_tools(self, status: ToolStatus = ToolStatus.ENABLED) -> List[Tool]:
        """retrieve tools by status, default get enabled."""
        return_tools = []
        for tool in self.tools_cfg:
            tool_cfg = self.tools_cfg[tool]
            if tool_cfg.status == status:
                return_tools.append(tool_cfg.function)
        return return_tools
    
    def get_tool(self, name: AvailableTools) -> Tool:
        """retrive a tool by its name"""
        return self.tools_cfg[name].function
    

# tools
@function_tool
async def _translate(text:str, src_lang:str, tar_lang:str) -> str:
    """Translate text from one to another. accept all language.

    Args:
        text: The text that need to be translated
        src_lang: The source language code. (e.g., 'vi' for Vietnamese)
        tar_lang: The target language code. (e.g., 'en' for English)

    Returns:
        A string of text that is translated from src_lang to tar_lang.
    """
    async with Translator() as translator:
        result = await translator.translate(
            text=text,
            src=src_lang,
            dest=tar_lang
        )
    logger.success("Đã sử dụng tool dịch.")
    return result.text


tools_manager = ToolsManager()
