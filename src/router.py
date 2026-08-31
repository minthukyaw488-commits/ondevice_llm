"""
Lightweight intent router for the welfare assistant.

Before running RAG, decide what kind of message this is:

  * smalltalk  - greetings / thanks / "how are you" / goodbye. Answer warmly
                 like a real chatbot; do NOT run retrieval or refuse.
  * welfare    - an actual welfare question -> RAG + grounded answer, but only
                 if retrieval is confident (see the relevance gate in pipeline).

This is rule-based (fast, local, no extra model). A message that mixes a
greeting with a real welfare topic ("안녕하세요, 기초연금 신청 방법 알려주세요")
is treated as welfare so the user still gets a real answer.
"""
from __future__ import annotations
import re

# Any welfare topic word -> this is a real question, not just chit-chat.
_WELFARE_HINT = re.compile(
    r"(연금|돌봄|급식|도시락|치매|응급|안전|건강|간호|혈압|당뇨|일자리|"
    r"바우처|난방|냉방|상담|신청|지원|복지|검진|지하철|경로|교통|우울|"
    r"외로|보건소|주민센터|서비스|어떻게|얼마|어디|받고\s*싶|알려)")

_GREETING = re.compile(r"(안녕|반갑|반가워|처음\s*뵙|좋은\s*아침|좋은\s*하루|하이|헬로)")
_THANKS = re.compile(r"(고마워|고맙|감사)")
_BYE = re.compile(r"(잘\s*있어|안녕히|다음에\s*봐|또\s*올게|이만|수고)")
_HOWRU = re.compile(
    r"(잘\s*지내|어떻게\s*지내|밥\s*(먹었|드셨)|식사\s*하셨|뭐\s*해|"
    r"이름이\s*뭐|누구세요|누구야|넌\s*누구|심심)")

_SMALLTALK = (_GREETING, _THANKS, _BYE, _HOWRU)


def is_smalltalk(text: str) -> bool:
    """True for greetings / thanks / light chit-chat with no welfare topic."""
    t = (text or "").strip()
    if not t:
        return False
    if _WELFARE_HINT.search(t):
        return False                      # has a real welfare topic -> answer it
    return any(p.search(t) for p in _SMALLTALK)


def classify(text: str) -> str:
    return "smalltalk" if is_smalltalk(text) else "welfare"
