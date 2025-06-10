import asyncio
from googletrans import Translator


translator_declaration = {
    "function_declarations": [{
            "name": "run_translate",
            "description": "Translate text from one language to another language. Use this when user asks to translate text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to translate from source language to target language",
                    },
                    "src_lang": {
                        "type": "string",
                        "description": "Source language code (2 letters, lowercase). Example: 'en' for English, 'vi' for Vietnamese, 'fr' for French, 'de' for German, 'ja' for Japanese, 'ko' for Korean, 'zh' for Chinese",
                    },
                        "tar_lang": {
                        "type": "string",
                        "description": "TTarget language code (2 letters, lowercase). Example: 'en' for English, 'vi' for Vietnamese, 'fr' for French, 'de' for German, 'ja' for Japanese, 'ko' for Korean, 'zh' for Chinese",
                    },
                },
                "required": ["text", "src_lang", "tar_lang"],
            },
    }]
}

async def translate(text:str, src_lang:str, tar_lang:str) -> str:
    async with Translator() as translator:
        result = await translator.translate(
            text=text,
            src=src_lang,
            dest=tar_lang
        )
    return result.text


def run_translate(text:str, src_lang:str, tar_lang:str) -> str:
    """Translate text.

    Args:
        text: Text to translate.translate from src_lang to tar_lang
        src_lang: Source Language. This is the orignal language of 'text'.
        tar_lang: Target Language. This is the target language to transte to.

    Returns:
        A string of text that is translated from src_lang to tar_lang.
    """
    return asyncio.run(translate(text=text, src_lang=src_lang, tar_lang=tar_lang))