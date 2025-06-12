import asyncio

from googletrans import Translator
from agents import function_tool

from _logger import logging
from _config import ToolStatus, Tools, ToolConfig

logger = logging.getLogger(__name__)

class ToolsManager:
    def __init__(self) -> None:
        self.tools_cfg = {
            Tools.TRANSLATE : ToolConfig(
                name="translate",
                status=ToolStatus.ENABLED,
                function=run_translate,
            )
        }

    def get_tools(self, status: ToolStatus = ToolStatus.ENABLED):
        """retrieve tools by status"""
        return_tools = []
        for tool in self.tools_cfg:
            tool_cfg = self.tools_cfg[tool]
            if tool_cfg.status == status:
                return_tools.append(tool_cfg.function)
        return return_tools
    
    def get_tool(self, name: Tools):
        """retrive a tool by its name"""
        return self.tools_cfg[name].function
    

# tools
async def _translate(text:str, src_lang:str, tar_lang:str) -> str:
    async with Translator() as translator:
        result = await translator.translate(
            text=text,
            src=src_lang,
            dest=tar_lang
        )
    logger.info("Đã sử dụng tool dịch.")
    return result.text


# declare
@function_tool
def run_translate(text:str, src_lang:str, tar_lang:str) -> str:
    """Translate text.

    Args:
        text: Text to translate.translate from src_lang to tar_lang
        src_lang: Source Language. This is the orignal language of 'text'.
        tar_lang: Target Language. This is the target language to transte to.

    Returns:
        A string of text that is translated from src_lang to tar_lang.
    """
    return asyncio.run(_translate(text=text, src_lang=src_lang, tar_lang=tar_lang))


tools_manager = ToolsManager()
