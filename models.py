import os

from openai import OpenAI
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key) # type: ignore

_embedding_model = None
_gemini_model = None
_private_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("huyydangg/DEk21_hcmute_embedding")
    return _embedding_model

def get_gemini_model():
    global _gemini_model
    if _gemini_model is None:
        _gemini_model = genai.GenerativeModel('gemini-2.0-flash') # type: ignore
    return _gemini_model

def get_private_model():
    global _private_model
    if _private_model is None:
        _private_model = OpenAI(
            base_url=os.getenv("PRIVATE_MODEL_URL"),
            api_key="no_key"
        )
    return _private_model