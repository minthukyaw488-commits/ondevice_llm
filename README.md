# 대전 독거노인 복지 안내 · 이상신호 감지 시스템
### On-device LLM + RAG for elderly welfare (Daejeon)

대전광역시 독거노인(61,527명)을 위한 **완전 로컬 실행** 음성 AI입니다.
어르신이 음성으로 복지 관련 질문을 하면 답변하고, 대화 속에서 반복되는
증상 호소나 정서적 이상신호를 감지해 **사회복지사에게 알림**을 보냅니다.

> **개인정보 보호 원칙**: 모든 처리는 로컬에서 실행됩니다.
> 대화 내용은 절대 외부(클라우드 API 등)로 전송되지 않습니다.

---

## 아키텍처 (2-layer)

```
어르신 음성 질문
    │
  STT (Whisper, 로컬)  →  텍스트
    │
    ├─▶ Layer 1: RAG 검색 (복지 문서 → chunk → 임베딩 → ChromaDB → 검색)
    │
    └─▶ Layer 2: 이상신호 분석 (증상 키워드 반복 + 감정 점수)
    │
  로컬 LLM (Ollama · llama3.1 / qwen2.5)
    │
  응답 생성  +  (이상신호 시) 사회복지사 알림 요약
```

## 파일 구성

| 파일 | 역할 |
|---|---|
| `src/config.py` | 모델·임계값 등 설정 |
| `src/rag_pipeline.py` | **Step 1** PDF 로딩(pdfplumber)·청킹·ChromaDB 검색 |
| `src/embeddings.py` | bge-m3 임베딩 (+ 오프라인 TF-IDF 대체) |
| `src/stt.py` | **Step 2** Whisper 음성인식 + 마이크 입력 |
| `src/abnormal_signal.py` | **Step 3** 감정 분석 + 증상 반복 추적 + 임계값 판정 |
| `src/llm.py` | 로컬 Ollama LLM 클라이언트 (클라우드 아님) |
| `src/pipeline.py` | **Step 4** 전체 통합 + 사회복지사 알림 생성 |
| `demo_cli.py` | **Step 5** 터미널 데모 |
| `app.py` | **Step 5** Streamlit 웹 데모 |
| `rag_minimal_example.py` | 최초 최소 프로토타입 (참고용) |
| `data/welfare_docs/` | 대전 복지 샘플 문서(대체 자료 10건) |

## 설치 & 실행

```bash
pip install -r requirements.txt

# 로컬 LLM (Ollama) - 클라우드 API 아님, 로컬 프로세스입니다.
# https://ollama.com 설치 후:
ollama pull llama3.1        # 또는 qwen2.5

# 터미널 데모
python demo_cli.py

# 웹 데모
streamlit run app.py

# 각 단계 개별 실행/확인
python -m src.rag_pipeline       # RAG 검색
python -m src.abnormal_signal    # 이상신호 감지
python -m src.pipeline           # 전체 파이프라인
```

## 이상신호 판정 로직 (rule-based)

`src/config.py`에서 조정할 수 있습니다.

- **증상 반복**: 같은 증상군(무릎/허리/수면/우울 등)을 `SYMPTOM_REPEAT_THRESHOLD`(기본 3)회 이상 호소 → 플래그
- **부정 감정**: 최근 대화 평균 부정 확률이 `NEGATIVE_SENTIMENT_THRESHOLD`(기본 0.6) 이상 → 플래그
- **위기 신호**: `CRISIS_KEYWORDS`("죽고 싶다" 등) 감지 시 → 즉시 긴급 알림
- 위 조건 중 하나라도 충족 시 **사회복지사 알림 요약** 자동 생성

## 모델 및 데이터

- **임베딩**: `BAAI/bge-m3` (다국어, 인증 불필요)
- **감정 분석**: 공개 한국어 sentiment 모델 (Hugging Face, 인증 불필요)
- **STT**: Whisper (`faster-whisper`)
- **LLM**: Ollama `llama3.1` / `qwen2.5` (로컬)
- **복지 문서**: 대전시 공개 복지정보 기반 **대체 샘플**. 실제 PDF 확보 시
  `data/welfare_docs/`에 넣으면 자동으로 색인됩니다.

## 오프라인 대체(fallback) 동작

각 구성요소는 모델을 내려받을 수 없는 환경에서도 파이프라인이 끊기지 않도록
**로컬 대체 경로**를 갖습니다. 실제 모델을 불러올 수 있는 환경에서는 자동으로
실제 모델을 사용합니다.

| 구성요소 | 기본(권장) | 대체 |
|---|---|---|
| 임베딩 | bge-m3 | TF-IDF (문자 n-gram) |
| 벡터 저장소 | ChromaDB | 인메모리 코사인 |
| 감정 분석 | HF 한국어 모델 | 한국어 감정 어휘 사전 |
| LLM | Ollama | 검색결과 요약 템플릿 |
| STT | Whisper | 텍스트 입력 안내 |

> 참고: 폐쇄망/일부 클라우드에서는 `huggingface.co`·`ollama.com` 접근이
> 차단될 수 있습니다. 그 경우 위 대체 경로로 데모가 동작하며,
> 인터넷이 되는 로컬 PC에서 최초 1회 모델을 받으면 실제 모델로 실행됩니다.
