import os

import google.generativeai as genai

from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key) # type: ignore

_embedding_model = None
_gemini_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
    return _embedding_model

def get_gemini_model():
    global _gemini_model
    if _gemini_model is None:
        _gemini_model = 