from pathlib import Path

from secret import *

CONFIG_PATH = Path(__file__).resolve()
ROOT_DIR    = CONFIG_PATH.parent
DB_PATH = str(ROOT_DIR / "data" / "chroma_db")
DATA_PATH = str(ROOT_DIR / "data" / "final_result.pkl")
SAVE_DIR = str(ROOT_DIR / "data" )

OPENAI_MODEL_NAME = "gpt-4o-mini"
OPENAI_EMBED_NAME = "text-embedding-3-small"

HIDDEN_TAG = "<당신의 AI 상담사>"  # 화면에 숨길 구분자
LOOKAHEAD  = len(HIDDEN_TAG)  # 태그 길이만큼 안전 버퍼