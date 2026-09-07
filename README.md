# 대전 독거노인 복지 안내 · 이상신호 감지 시스템
### On-device LLM + RAG for elderly welfare (Daejeon)

대전광역시 독거노인을 위한 **완전 로컬 실행** 음성·문자 AI입니다.
어르신이 음성 또는 글로 복지 관련 질문을 하면 **문서에 근거해 답변**하고,
대화 속 반복 증상 호소나 정서적 이상신호를 감지해 **사회복지사에게 알림**을
보냅니다.

> **개인정보 보호 원칙**: LLM·검색·재랭킹·음성인식 등 모든 처리가 기기 안에서
> 실행됩니다. 대화 내용은 절대 외부(클라우드 API 등)로 전송되지 않습니다.

---

## 개발 진행 현황 (Roadmap)

| 단계 | 내용 | 상태 |
|---|---|---|
| **1. 텍스트 대화** | RAG 기반 복지 Q&A (정제·재랭킹·관련성 게이트·의도 라우팅) | ✅ **완료** |
| **2. 이상신호 감지** | 감정 분석 + 증상 반복 + 개인 baseline 지표 | ✅ 구현 (고도화 예정) |
| **3. 두 모델 융합** | 하나의 입력을 ①복지 안내 + ②이상신호로 동시 처리 | ✅ 통합 완료 |
| 4. 음성 대화 | STT·TTS voice-to-voice (핸즈프리) | 🔷 실험 구현 |
| 5. UI 디자인 | 정부포털 스타일 · 접근성 · 모바일/PWA | ✅ 구현 |
| 6. 평가 | 정량 평가 하네스 (RAG·이상신호·답변품질·모델비교) | ✅ 구축 |

**1단계(텍스트 대화) 성능:** 실데이터 기반 200문항 평가에서 **정답률 약 92%**
(정보 질문 89~91% / **범위 밖 안전 거절 100%**), **사실 정확도 98%** · 한국어 준수 98%
· 평균 응답 4~6초. 재랭킹 적용 시 **+14%p** 향상.
(온디바이스 2.4B 모델 특성상 실행마다 ±2~3%p 변동, 범위 밖 거절과 사실 정확도는 안정적.)

---

## 시스템 아키텍처 (2-layer)

```
어르신 입력 (음성 또는 문자)
    │
  STT (Whisper, 로컬)  →  텍스트
    │
  의도 라우팅 ── 인사·잡담 ─▶ 따뜻한 대화형 응답 (RAG 미사용)
    │  └ 복지 질문
    ├─▶ Layer 1: RAG  (정제 → 섹션 청킹 → bge-m3 임베딩 → ChromaDB 후보 10
    │              → cross-encoder 재랭킹 → 관련성 게이트 → 상위 3)
    │
    └─▶ Layer 2: 이상신호 분석 (증상 반복 + 감정 점수 + 개인 baseline)
    │
  로컬 LLM (Ollama · EXAONE 3.5 2.4B) — 검색 근거만으로 한국어 답변
    │
  응답 생성  +  (이상신호 시) 사회복지사 알림 요약
```

**핵심 설계**
- **개인정보 보호**: 전 구간 로컬. 대화가 기기/가정 네트워크를 벗어나지 않음.
- **안정성**: 모델 추론을 Streamlit과 분리된 별도 워커 프로세스에서 실행(bus error 방지).
- **검색 정밀도**: 임베딩으로 후보를 넓게 뽑고 cross-encoder로 재랭킹.
- **환각 억제**: 관련성 게이트 — 검색 점수가 낮으면 잘못된 근거로 답하지 않고 주민센터 안내.
- **자연스러운 대화**: 인사/잡담은 라우터가 따뜻하게 응대.
- **off-domain 가드**: 복지와 무관한 질문(투자·반려동물·여행·가전 등)은 검색 전에
  걸러 곧바로 안전 안내 → 대규모 시설 데이터에서도 환각 없이 100% 거절.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `src/config.py` | 모델·임계값 등 설정 (LLM/임베딩/재랭킹/게이트) |
| `src/rag_pipeline.py` | **Step 1** 문서 로딩(PDF/MD/TXT/CSV/XLS/XLSX)·정제·청킹·재랭킹·검색 |
| `src/reranker.py` | Cross-encoder 재랭킹 (BAAI/bge-reranker-v2-m3) |
| `src/router.py` | 의도 라우팅 (인사·잡담 vs 복지 질문) |
| `src/embeddings.py` | bge-m3 임베딩 (+ 오프라인 TF-IDF 대체) |
| `src/stt.py` | **Step 2** Whisper 음성인식 |
| `src/abnormal_signal.py` | **Step 3** 감정 분석 + 증상 반복 + baseline 이상징후 |
| `src/llm.py` | 로컬 Ollama LLM 클라이언트 + 프롬프트(근거 기반·잡담·거절) |
| `src/pipeline.py` | **Step 4** 전체 통합 (라우팅·게이트·알림) |
| `src/worker.py`, `src/worker_server.py` | 모델 추론용 별도 워커 프로세스 |
| `src/vad_stream.py`, `src/tts.py` | 핸즈프리 음성(VAD) · 로컬 TTS |
| `app.py` | Streamlit 웹/모바일 데모 (정부포털 스타일, PWA) |
| `desktop_app.py` | 데스크톱 앱(전체화면) 실행기 |
| `run_phone.sh` | 휴대폰용 로컬 HTTPS 실행기 (마이크 지원) |
| `set_hero_image.py` | 홈 배너 배경 이미지 설정 |
| `evaluate.py` | RAG Hit@k · 이상신호 F1 평가 |
| `eval_qa.py` | 답변 품질 평가 (200문항, `data/eval/qa_200.jsonl`) |
| `eval_models.py` | 여러 로컬 LLM 성능 비교 |
| `make_eval200.py` | 200문항 평가셋 생성기 |
| `data/welfare_docs/` | 복지 문서 (실데이터 + 샘플) |

## 설치 & 실행

```bash
pip install -r requirements.txt

# 로컬 LLM (Ollama) — 클라우드 API 아님, 로컬 프로세스입니다. https://ollama.com
ollama pull exaone3.5:2.4b        # 기본 모델 (한국어 특화, LG AI)
#  또는 qwen2.5:1.5b  등 — 모델은 자동 감지/대체됩니다.

# 문서 색인 확인 (RAG 검색)
python -m src.rag_pipeline

# 웹/모바일 데모 (Streamlit) — 모델이 설치된 동일 환경의 파이썬으로 실행.
python -m streamlit run app.py

# 데스크톱 앱 (태블릿·PC·미니PC)
python desktop_app.py

# 답변 품질 평가 (200문항)
python eval_qa.py
```

> `streamlit run`이 다른 환경(anaconda base 등)을 가리키면 워커가 그 환경의
> torch를 로드하다 bus error로 죽을 수 있습니다. 모델이 설치된 환경에서
> `python -m streamlit run app.py`로 실행하거나, `WORKER_PYTHON`에 그 환경의
> 파이썬 경로를 지정하세요.

## 데이터

복지 문서를 `data/welfare_docs/`에 넣으면 **자동으로 색인**됩니다. 지원 형식:

| 형식 | 처리 |
|---|---|
| `.pdf` | pdfplumber 추출 + 스캔 페이지 OCR(tesseract-kor) |
| `.md`, `.txt` | 직접 로딩 |
| `.csv` | 행을 "컬럼: 값, …" 텍스트로 변환 (cp949/euc-kr 지원) |
| `.xls`, `.xlsx` | pandas로 로딩 (openpyxl/xlrd 필요) |

문서는 **정제(머리말/꼬리말·페이지번호·기호 잡음 제거) → 섹션 단위 청킹 →
저품질 청크 제거** 후 색인됩니다. (`.hwp`는 아직 미지원 — 향후 과제)

**현재 사용 데이터(예)**: 보건복지부 「2025 노인보건복지 사업안내(1·2권)」,
대전광역시 노인맞춤돌봄·재가노인 식사배달 현황(CSV), 실버 시설 현황(XLS) 등.

## 이상신호 판정 로직 (rule-based)

`src/config.py`에서 조정할 수 있습니다.

- **증상 반복**: 같은 증상군(무릎/허리/수면/우울 등)을 `SYMPTOM_REPEAT_THRESHOLD`(기본 3)회 이상 → 플래그
- **부정 감정**: 최근 대화 평균 부정 확률이 `NEGATIVE_SENTIMENT_THRESHOLD`(기본 0.6) 이상 → 플래그
- **위기 신호**: `CRISIS_KEYWORDS`("죽고 싶다" 등) 감지 시 → 즉시 긴급 알림
- **이상징후(baseline 대비 변화)**: 최근(3일) vs 평소(14일) 구간을 비교해 대화 빈도
  감소·부정감정 상승·외로움 표현 증가·응답 길이 위축 4지표로 정량화

## 평가 (Evaluation)

```bash
python evaluate.py       # RAG Hit@1/Hit@3, 이상신호 정확도/정밀도/재현율/F1
python eval_qa.py        # 답변 품질 200문항 (정확성·한국어·간결성·거절)
python eval_models.py exaone3.5:2.4b qwen2.5:1.5b   # 모델 비교
```

**답변 품질 (실데이터, 200문항, EXAONE 3.5 2.4B, 재랭킹 ON)**

| 지표 | 결과 |
|---|---|
| 전체 정답률(pass) | **약 92%** (183~186/200) |
| 정보 질문(정답) | 89~91% (133~136/150) |
| 범위 밖(안전한 거절) | **100%** (50/50) |
| 사실 정확도(fact hit) | **98%** |
| 한국어 준수 | 98% · 간결성 96% · 평균 응답 4~6초 |

- **범위 밖 거절 100%**: 의도 라우터의 off-domain 가드가 복지와 무관한 질문
  (투자·반려동물·여행·가전 등)을 검색 전에 걸러 환각을 원천 차단.
- **재랭킹 효과**: 재랭킹 적용 시 정답률 **+14%p** 향상(잘못된 청크 상위노출 완화).
- 남은 오답은 대부분 답변이 길거나(간결성) 시설 목록을 함께 안내해 기대 키워드가
  빠진 경우로, 사실 정확도(fact hit)는 **97%**로 높음.
- **모델 비교**: 실데이터에서 한국어 특화 **EXAONE 3.5 > Qwen 2.5** (Llama 계열은 영어 혼입).
- 한계: 평가셋은 자체 제작(held-out 미구성), 지표는 키워드 기반이므로 향후 사람/LLM 평가로 보강 예정.

## 휴대폰에서 사용 (모바일 웹 · PWA)

LLM·음성은 집/센터의 허브(미니PC·태블릿)에서 로컬 실행하고, 휴대폰은 같은
WiFi로 접속하는 "화면·마이크" 역할만 합니다.

```bash
# 허브에서 실행 (같은 WiFi의 휴대폰이 접속)
python -m streamlit run app.py --server.address 0.0.0.0

# 음성(마이크)까지 쓰려면 로컬 HTTPS 실행기 (모두 로컬, 클라우드·터널 없음)
./run_phone.sh
```
휴대폰 브라우저에서 접속 후 **홈 화면에 추가**하면 앱처럼 전체화면으로 열립니다.

## 모델 및 데이터

- **LLM**: Ollama `exaone3.5:2.4b` (기본, 한국어 특화) — `qwen2.5` 등 자동 대체
- **임베딩**: `BAAI/bge-m3` (다국어, 인증 불필요)
- **재랭킹**: `BAAI/bge-reranker-v2-m3` (cross-encoder)
- **STT**: Whisper (`faster-whisper`)
- **감정 분석**: 공개 한국어 sentiment 모델 (Hugging Face)

각 구성요소는 모델을 내려받을 수 없는 환경에서도 파이프라인이 끊기지 않도록
로컬 대체 경로(TF-IDF 임베딩, 인메모리 벡터, 감정 어휘 사전, 검색결과 요약)를
갖습니다.

## 참고문헌 (모델·기법 근거)

- Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. arXiv:2005.11401
- Chen et al. *M3-Embedding (BGE-M3).* ACL 2024 Findings. arXiv:2402.03216
- Nogueira & Cho. *Passage Re-ranking with BERT.* arXiv:1901.04085
- LG AI Research. *EXAONE 3.5.* arXiv:2412.04862
- Qwen Team. *Qwen2.5 Technical Report.* arXiv:2412.15115
- Son et al. *KMMLU: Measuring Massive Multitask Language Understanding in Korean.* NAACL 2025. arXiv:2402.11548
- Radford et al. *Robust Speech Recognition via Large-Scale Weak Supervision (Whisper).* ICML 2023. arXiv:2212.04356
