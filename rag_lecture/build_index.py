"""
교수님 강의(Week 05 · STEP 09) 방식의 가장 단순한 RAG — 우리 팀 데이터(대전 복지)로.
목요일 과제: "팀 주제 문서로 바꿔보기"를 그대로 구현한 것입니다.

강의와 동일한 뼈대 → 찾고 · 넣고 · 답한다
  · 임베딩 모델 : Ollama bge-m3  (/api/embeddings)  ← 강의와 동일
  · 조각내기    : 글자 500자, 50자 겹침            ← 강의와 동일
  · 저장        : vectors.npy + chunks.json        ← 가장 단순한 벡터 DB

이 파일(build_index.py)은 한 번만 실행합니다. data/welfare_docs 의 문서를
읽어 조각내고, 한 번 임베딩해서 파일로 저장합니다.

  pip install requests numpy --break-system-packages   # 강의 slide 57
  ollama pull bge-m3                                    # 한국어 임베딩 모델
  python rag_lecture/build_index.py
"""
import glob
import json
import os

import numpy as np
import requests

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")   # 안되면 nomic-embed-text
DOCS_DIR = os.environ.get("DOCS_DIR", "data/welfare_docs")
HERE = os.path.dirname(os.path.abspath(__file__))


def embed(text: str):
    """문장을 숫자 배열(벡터)로 — 강의와 동일하게 Ollama 임베딩 API 사용."""
    r = requests.post(OLLAMA + "/api/embeddings",
                      json={"model": EMBED_MODEL, "prompt": text})
    r.raise_for_status()
    return r.json()["embedding"]


def chunk(text: str, size: int = 500, overlap: int = 50):
    """글자 기준으로 자르되 앞뒤 50자를 겹친다 — 강의 slide 64와 동일."""
    out, i = [], 0
    while i < len(text):
        piece = text[i:i + size].strip()
        if piece:
            out.append(piece)
        i += size - overlap
    return out


def read_file(path: str) -> str:
    """대전 복지 문서를 텍스트로. txt/md/csv는 그대로, pdf는 글자 추출."""
    ext = path.lower().rsplit(".", 1)[-1]
    if ext in ("txt", "md", "csv"):
        for enc in ("utf-8", "cp949", "euc-kr"):
            try:
                return open(path, encoding=enc).read()
            except UnicodeDecodeError:
                continue
        return open(path, encoding="utf-8", errors="ignore").read()
    if ext == "pdf":
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        except Exception as e:
            print(f"  (pdf 건너뜀: {os.path.basename(path)} — {e})")
    return ""


def main():
    paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*")))
    print(f"문서 폴더: {DOCS_DIR}  ({len(paths)}개 파일)\n")

    chunks, sources = [], []
    for p in paths:
        text = read_file(p)
        if not text.strip():
            continue
        cs = chunk(text)
        chunks += cs
        sources += [os.path.basename(p)] * len(cs)   # 근거 표시용 출처
        print(f"  {os.path.basename(p):42} {len(cs):4} 조각")

    print(f"\n총 조각 수: {len(chunks)}")
    if not chunks:
        print(f"문서를 찾지 못했습니다. DOCS_DIR 를 확인하세요: {DOCS_DIR}")
        return

    print("임베딩 시작 (조각마다 한 번씩 — nvidia-smi 로 GPU 가동 확인):")
    vecs = []
    for i, c in enumerate(chunks):
        vecs.append(embed(c))
        if i % 50 == 0:
            print(f"  {i}/{len(chunks)}")

    np.save(os.path.join(HERE, "vectors.npy"), np.array(vecs))
    json.dump(chunks, open(os.path.join(HERE, "chunks.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    json.dump(sources, open(os.path.join(HERE, "sources.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"\n저장 완료 → vectors.npy  chunks.json  sources.json  ({len(chunks)} 조각)")


if __name__ == "__main__":
    main()
