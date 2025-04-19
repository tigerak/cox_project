from pathlib import Path

from secret import *

CONFIG_PATH = Path(__file__).resolve()
ROOT_DIR    = CONFIG_PATH.parent

# Data
DATA_PATH = str(ROOT_DIR / "data" / "final_result.pkl")
SAVE_DIR = str(ROOT_DIR / "data" )

# OpenAI 
OPENAI_MODEL_NAME = "gpt-4o-mini"
OPENAI_EMBED_NAME = "text-embedding-3-small"

# ChromaDB
DB_NAME = "consult_manual"
DB_PATH = str(ROOT_DIR / "data" / "chroma_db")

# GPT-mini Respons
HIDDEN_TAG = "<당신의 AI 상담사>"  # 화면에 숨길 구분자
LOOKAHEAD  = len(HIDDEN_TAG)  # 태그 길이만큼 안전 버퍼

# Conditional search
ALPHA = 0.8