"""
Central configuration for the on-device elderly welfare voice AI system.

All processing runs locally. No cloud LLM / API calls are allowed anywhere
in this project (privacy requirement: conversation data of 독거노인 must never
leave the device).
"""
from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
WELFARE_DOCS_DIR = DATA_DIR / "welfare_docs"
AUDIO_DIR = DATA_DIR / "audio"
CHROMA_DIR = DATA_DIR / "chroma_db"          # persisted vector store

# --- RAG / embedding -----------------------------------------------------
# bge-m3 is multilingual (Korean + English) and was validated in the
# original prototype. If sentence-transformers or the model is unavailable,
# the pipeline falls back to a lightweight local embedding (see embeddings.py)
# so the demo still runs offline.
EMBED_MODEL = "BAAI/bge-m3"
CHROMA_COLLECTION = "daejeon_welfare"

# Chunking: welfare PDFs can be long, so split into small overlapping chunks.
CHUNK_SIZE = 400          # target characters per chunk (200-500 range)
CHUNK_OVERLAP = 80        # characters shared between neighbouring chunks
RAG_TOP_K = 3             # how many chunks to retrieve per question

# --- Local LLM (Ollama) --------------------------------------------------
# Runs fully on-device via the Ollama server (http://localhost:11434).
# This is NOT a cloud API - it is a local process. If Ollama is not running,
# the pipeline uses a transparent template fallback so the rest of the
# system can still be demonstrated.
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "llama3.1"    # or "qwen2.5" / a smaller tag like "qwen2.5:0.5b"
LLM_TIMEOUT = 120         # seconds

# --- STT (Whisper) -------------------------------------------------------
WHISPER_MODEL = "base"    # tiny/base/small - base is a good speed/quality mix
WHISPER_LANGUAGE = "ko"   # elderly speak Korean

# --- Abnormal signal detection ------------------------------------------
# Public Korean sentiment model (no auth / token needed).
SENTIMENT_MODEL = "sangrimlee/bert-base-multilingual-cased-nsmc"
# Fallback smaller multilingual option is handled in abnormal_signal.py.

# Symptom keywords to track across the conversation history. Grouped so that
# different phrasings of the same complaint count together.
SYMPTOM_KEYWORDS = {
    "무릎": ["무릎", "무릎이", "무릎 아프", "다리 아프", "다리가 아프"],
    "허리": ["허리", "허리가 아프", "허리 아프"],
    "두통": ["머리 아프", "머리가 아프", "두통", "어지러"],
    "가슴": ["가슴이 아프", "가슴 답답", "숨이 차", "숨쉬기"],
    "수면": ["잠을 못", "잠이 안", "불면", "밤에 못 자"],
    "외로움": ["외로", "혼자", "쓸쓸", "말할 사람이 없"],
    "우울": ["죽고 싶", "살기 싫", "우울", "의욕이 없", "희망이 없"],
    "식사": ["밥을 못", "입맛이 없", "먹기 싫", "굶"],
}

# Thresholds for the rule-based abnormal-signal decision.
SYMPTOM_REPEAT_THRESHOLD = 3      # same symptom mentioned >= N times -> flag
NEGATIVE_SENTIMENT_THRESHOLD = 0.6  # avg negative probability -> flag
# The sentiment model is binary (no neutral class), so a single info-seeking
# question can score as negative. Require a minimum number of utterances
# before the average-sentiment rule may fire, to avoid false alarms.
SENTIMENT_MIN_UTTERANCES = 3
# High-risk keywords escalate immediately regardless of counts.
CRISIS_KEYWORDS = ["죽고 싶", "살기 싫", "자살", "죽어야"]
