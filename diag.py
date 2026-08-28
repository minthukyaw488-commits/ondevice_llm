"""
Bus-error diagnostic. Run in plain Python (main thread):

    python diag.py

Bus errors give no Python traceback, so we print each step and flush. The
LAST line printed before the crash tells us which native library is at fault.
Send the full output back.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import platform
def step(msg): print(msg, flush=True)

step(f"python {platform.python_version()} | {platform.machine()} | {platform.platform()}")

step("1) import numpy");            import numpy; step(f"   numpy {numpy.__version__}")
step("2) import torch");            import torch; step(f"   torch {torch.__version__}")
step("3) import chromadb");         import chromadb; step("   chromadb ok")
step("4) import sentence_transformers"); import sentence_transformers; step("   ok")
step("5) load bge-m3 (downloads first time)")
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-m3"); step("   bge-m3 loaded")
step("6) encode test"); v = m.encode(["기초연금 신청"]); step(f"   vector dim {len(v[0])}")
step("7) import transformers pipeline"); from transformers import pipeline; step("   ok")
step("8) load Korean sentiment model")
p = pipeline("sentiment-analysis", model="sangrimlee/bert-base-multilingual-cased-nsmc")
step(f"   sentiment: {p('오늘 너무 힘들어요')}")
step("9) build full assistant (chromadb + models together)")
from src.pipeline import WelfareAssistant
b = WelfareAssistant(); step("   assistant built")
step("ALL OK — no crash in plain Python")
