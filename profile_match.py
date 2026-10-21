"""
사람 맞춤형 MVP 데모 — 어르신 프로필을 받아 '받을 수 있는 복지'를 추천한다.

RAG(질문 맞춤형)의 다음 단계인 '사람 맞춤형'을 가장 단순하게 보여주는 데모.
외부 라이브러리·LLM 없이 표준 라이브러리만으로 동작한다.

  python profile_match.py            # 미리 준비한 예시 프로필 3명 출력
  python profile_match.py --ask      # 직접 입력(대화형)
  python profile_match.py --rag      # 추천 복지마다 실제 문서 근거(RAG)까지 표시

--rag 는 rag_lecture 인덱스(vectors.npy)와 Ollama 임베딩이 있을 때만 동작하며,
없으면 자동으로 규칙 기반 결과만 보여준다(어디서나 데모 가능).

※ 자격 기준은 대략치이며, 정확한 금액·등급은 공공문서(RAG)·주민센터로 확인.
"""
from __future__ import annotations
import os
import sys

from src.welfare_rules import Profile, Welfare, recommend, DAEJEON_DISTRICTS

USE_RAG = "--rag" in sys.argv
# 근거로 보여줄 조각 수
EVIDENCE_K = int(os.environ.get("RAG_EVIDENCE_K", "2"))


def _load_rag():
    """rag_lecture/ask.py 를 안전하게 로드. 인덱스가 없으면 None(근거 생략)."""
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    ask_path = os.path.join(here, "rag_lecture", "ask.py")
    try:
        spec = importlib.util.spec_from_file_location("rag_lecture_ask", ask_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)   # vectors.npy 없으면 여기서 예외
        return mod
    except Exception as e:                # noqa: BLE001 — 데모는 조용히 규칙 기반으로
        print(f"  (RAG 근거 생략 — 인덱스/임베딩 불가: {type(e).__name__})")
        return None


# 검색 실패 원인을 한 번만 보여주기 위한 플래그
_RAG_ERR_SHOWN = False


def _evidence_for(rag, w: Welfare):
    """복지 하나에 대한 실제 문서 근거 조각 목록. 실패 시 빈 리스트.
    첫 실패의 원인(예: Ollama 미실행, 임베딩 모델 없음)은 한 번 출력한다."""
    global _RAG_ERR_SHOWN
    query = f"{w.name} 지원 대상 신청 자격"
    try:
        return rag.search(query, k=EVIDENCE_K)   # [(chunk, source, sim), ...]
    except Exception as e:                        # noqa: BLE001
        if not _RAG_ERR_SHOWN:
            print(f"  ⚠ RAG 검색 실패: {type(e).__name__}: {e}")
            print("    (Ollama 실행 여부·임베딩 모델(bge-m3)·vectors.npy 확인)")
            _RAG_ERR_SHOWN = True
        return []


def _print_result(p: Profile, rag=None):
    eligible, maybe, not_elig = recommend(p)

    print("\n" + "=" * 60)
    print(" 입력 프로필")
    print("=" * 60)
    print(f"  나이        : {p.age if p.age is not None else '-'}")
    print(f"  저소득      : {_yn(p.income_low)}")
    print(f"  거주 구     : {p.district or '-'}")
    print(f"  독거 여부   : {_yn(p.living_alone)}")
    print(f"  거동 불편   : {_yn(p.mobility_issue)}")
    print(f"  만성질환    : {_yn(p.chronic_illness)}")

    print("\n  ✅ 받을 수 있는 복지")
    if eligible:
        for r in eligible:
            print(f"     • {r.welfare.name} — {r.welfare.summary}")
            for why in r.reasons:
                print(f"         - {why}")
            print(f"         ※ {r.welfare.verify_note}")
            if rag is not None:
                hits = _evidence_for(rag, r.welfare)
                if hits:
                    print("         📄 문서 근거(RAG):")
                    for chunk, src, sim in hits:
                        snippet = " ".join(chunk.split())[:90]
                        print(f"            [{sim:.3f}] ({src or '출처미상'}) {snippet}…")
                else:
                    print("         📄 문서 근거: 관련 조각을 찾지 못함")
    else:
        print("     (해당 없음 — 아래 '확인 필요' 참고)")

    if maybe:
        print("\n  ❓ 확인하면 받을 수 있음 (정보 부족)")
        for r in maybe:
            need = [w for w in r.reasons if "정보 필요" in w]
            print(f"     • {r.welfare.name} — {', '.join(need) if need else '추가 확인'}")

    if not_elig:
        print("\n  ✖ 현재 조건상 대상 아님")
        for r in not_elig:
            print(f"     • {r.welfare.name} — {r.reasons[0] if r.reasons else ''}")

    print("\n  → 정확한 신청·금액은 주민센터 또는 복지 상담(RAG)으로 확인하세요.")


def _yn(v):
    return "예" if v is True else ("아니오" if v is False else "모름")


# ── 미리 준비한 예시 프로필(발표 데모용) ──────────────────────────
SAMPLES = [
    ("저소득 독거·거동불편 어르신 (서구)",
     Profile(age=72, income_low=True, district="서구",
             living_alone=True, mobility_issue=True, chronic_illness=False)),
    ("중간소득·배우자 동거 어르신 (유성구)",
     Profile(age=68, income_low=False, district="유성구",
             living_alone=False, mobility_issue=False, chronic_illness=True)),
    ("60대 초반 저소득 독거 어르신 (동구)",
     Profile(age=62, income_low=True, district="동구",
             living_alone=True, mobility_issue=False, chronic_illness=False)),
]


def _ask_bool(prompt: str):
    v = input(prompt + " (y/n, 모르면 엔터): ").strip().lower()
    if v in ("y", "예", "네", "응"):
        return True
    if v in ("n", "아니오", "아니"):
        return False
    return None


def _ask_int(prompt: str):
    v = input(prompt + " (모르면 엔터): ").strip()
    try:
        return int(v)
    except ValueError:
        return None


def ask_profile() -> Profile:
    print("\n어르신 정보를 입력해 주세요 (모르면 엔터로 건너뜀).")
    age = _ask_int("  나이(만)")
    income_low = _ask_bool("  저소득(기초수급·차상위 등)입니까?")
    print(f"  거주 구 {DAEJEON_DISTRICTS} 중 하나")
    district = input("  거주 구 (모르면 엔터): ").strip() or None
    living_alone = _ask_bool("  혼자 사십니까(독거)?")
    mobility_issue = _ask_bool("  거동이 불편하십니까?")
    chronic_illness = _ask_bool("  만성질환·기억력 저하가 있으십니까?")
    return Profile(age=age, income_low=income_low, district=district,
                   living_alone=living_alone, mobility_issue=mobility_issue,
                   chronic_illness=chronic_illness)


def main():
    rag = _load_rag() if USE_RAG else None
    if rag is not None:
        print("  (RAG 근거 표시 ON — rag_lecture 인덱스 사용)")

    if "--ask" in sys.argv:
        _print_result(ask_profile(), rag=rag)
        return
    print("사람 맞춤형 복지 추천 데모 — 예시 프로필 3명\n(직접 입력: python profile_match.py --ask)")
    for title, p in SAMPLES:
        print(f"\n\n### {title}")
        _print_result(p, rag=rag)


if __name__ == "__main__":
    main()
