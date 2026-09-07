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

# Actual welfare TOPIC words (not generic question words like 어떻게/어디). Used
# only by the off-domain guard: if a real welfare topic is present we never
# treat the message as off-domain, even if it also contains an off-topic word
# (e.g. "장기요양보험 신청" contains 요양 -> welfare, not off-domain).
_WELFARE_TOPIC = re.compile(
    r"(연금|돌봄|급식|도시락|반찬|치매|응급|안전|간호|혈압|당뇨|복약|건강관리|"
    r"일자리|바우처|난방|냉방|정신건강|우울|외로|고독|보건소|주민센터|행정복지|"
    r"경로당|경로우대|무임|복지관|요양|재가|생활지원|방문건강|어르신|노인|독거|"
    r"기초연금|장기요양|돌봄서비스|안심서비스)")

# Clearly non-welfare topics. If one appears and there is NO welfare topic, the
# question is out-of-domain: decline it WITHOUT retrieval so a large facility
# index cannot surface a spurious match for the model to hallucinate around.
_OFF_DOMAIN = re.compile(
    r"(주식|투자|부동산|비트코인|코인|대출|이자|신용카드|보험|환율|집값|주가|"
    r"비행기|항공권|해외여행|여행지|여행|호텔|숙박|"
    r"스마트폰|핸드폰|휴대폰|노트북|컴퓨터\s*수리|세탁기|냉장고|가전|인터넷|와이파이|"
    r"동물병원|강아지|고양이|반려동물|"
    r"자동차|중고차|운전면허|기차표|기차|KTX|"
    r"로또|복권|"
    r"맛집|김치찌개|탕수육|레시피|요리\s*법|커피|"
    r"날씨|축구|야구|농구|드라마|영화|게임|연예인|아이돌|유튜브|넷플릭스|"
    r"선거|대통령|번역|수학\s*문제|택배|화장품|축의금|부업|입시|이력서|"
    r"코딩|프로그래밍|다이아몬드|반지|살\s*빼|다이어트)")


def is_off_domain(text: str) -> bool:
    """True for a clearly non-welfare request (invest/pets/travel/gadgets/...).

    Guarded by _WELFARE_TOPIC: a message that mentions a real welfare topic is
    never off-domain. Generic question words (어떻게/어디/신청) do NOT count as
    welfare here, so "주식 투자는 어떻게 시작하나요?" is still off-domain.
    """
    t = (text or "").strip()
    if not t or _WELFARE_TOPIC.search(t):
        return False
    return bool(_OFF_DOMAIN.search(t))


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
