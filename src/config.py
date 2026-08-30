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
ALERT_LOG_DIR = DATA_DIR / "alerts"          # local social-worker alert log

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
RAG_TOP_K = 2             # fewer chunks -> shorter prompt -> faster LLM reply

# --- Local LLM (Ollama) --------------------------------------------------
# Runs fully on-device via the Ollama server (http://localhost:11434).
# This is NOT a cloud API - it is a local process. If Ollama is not running,
# the pipeline uses a transparent template fallback so the rest of the
# system can still be demonstrated.
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "exaone3.5:2.4b"  # Korean-native (LG AI); auto-falls back if absent
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

# --- Anomaly detection: deviation from the person's own baseline ----------
# Anomaly = a meaningful change from an individual's normal pattern, measured
# by comparing a RECENT window against a BASELINE window. This needs multi-day
# history; in a single short session these metrics stay inactive.
RECENT_WINDOW_DAYS = 3        # "recent" behaviour window
BASELINE_WINDOW_DAYS = 14     # "normal/baseline" window (includes recent span)
MIN_BASELINE_UTTERANCES = 5   # need at least this much history to trust a baseline

# Emotional-distress keywords whose *emergence/increase* signals a mood shift.
EMOTION_KEYWORDS = ["외로", "쓸쓸", "혼자", "슬프", "우울", "그립", "보고 싶",
                    "눈물", "허전", "재미가 없", "의욕이 없", "희망이 없"]

# Thresholds for each baseline-vs-recent metric.
FREQ_DROP_THRESHOLD = 0.5         # recent chat rate <= 50% of baseline -> flag
SENTIMENT_SHIFT_THRESHOLD = 0.25  # recent avg negativity rose by >= this
KEYWORD_SHIFT_THRESHOLD = 0.30    # emotional-keyword rate rose by >= this
LENGTH_DROP_THRESHOLD = 0.5       # recent avg reply length <= 50% of baseline
