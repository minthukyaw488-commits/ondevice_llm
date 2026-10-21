"""
대전노인복지 HUB — 사람 맞춤형 통합 웹앱 (로그인 → 프로필 → 맞춤 상담).

일반 챗봇은 누구에게나 같은 일반 정보를 주지만, 이 앱은 어르신이 '내 계정'에
자기 정보를 넣어두면, 로그인하는 순간 '당신이 받을 수 있는 복지'를 알려주고,
이어지는 대화도 그 어르신의 나이·소득·거주지에 맞춰 답한다.

세 가지를 하나로 합쳤다:
  1) 계정(이름 기반, 프로토타입)         — 어르신별 프로필 저장
  2) 자격 매칭(src.welfare_rules)        — 받을 수 있는 복지 자동 추천
  3) RAG 상담(rag_lecture) + 프로필 주입  — 그 사람 상황에 맞춘 문서 근거 답변

설치·실행:
  pip install fastapi uvicorn
  python3 -m uvicorn app_personal:app --host 0.0.0.0 --port 8002
  브라우저:  http://localhost:8002

※ 프로토타입이라 비밀번호 없이 이름만으로 계정을 구분한다. 실제 서비스에서는
  본인 동의·인증·암호화가 필요하다(개인정보). 프로필은 data/profiles.json에
  로컬 저장하며 git에 커밋하지 않는다.
"""
from __future__ import annotations
import json
import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.welfare_rules import Profile, recommend

app = FastAPI(title="대전노인복지 HUB — 맞춤 상담")

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "data", "profiles.json")
EVIDENCE_K = int(os.environ.get("RAG_EVIDENCE_K", "3"))
EVIDENCE_MIN_SIM = float(os.environ.get("RAG_EVIDENCE_MIN_SIM", "0.62"))

# ── 아주 단순한 계정 저장소(이름 → 프로필) ──────────────────────────
def _load_store() -> dict:
    try:
        return json.load(open(STORE, encoding="utf-8"))
    except Exception:                      # noqa: BLE001
        return {}


def _save_store(d: dict):
    os.makedirs(os.path.dirname(STORE), exist_ok=True)
    json.dump(d, open(STORE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


# ── RAG(rag_lecture/ask.py) 지연 로드 ──────────────────────────────
_RAG = None
_RAG_TRIED = False


def _get_rag():
    global _RAG, _RAG_TRIED
    if _RAG_TRIED:
        return _RAG
    _RAG_TRIED = True
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location(
            "rag_lecture_ask", os.path.join(HERE, "rag_lecture", "ask.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _RAG = mod
    except Exception:                      # noqa: BLE001
        _RAG = None
    return _RAG


def _profile_from(d: dict) -> Profile:
    return Profile(
        age=d.get("age"), income_low=d.get("income_low"),
        district=d.get("district"), living_alone=d.get("living_alone"),
        mobility_issue=d.get("mobility_issue"), chronic_illness=d.get("chronic_illness"))


def _profile_text(d: dict) -> str:
    """프롬프트에 넣을 어르신 정보 한 줄 요약."""
    def yn(v):
        return "예" if v is True else ("아니오" if v is False else "모름")
    parts = []
    if d.get("age") is not None:
        parts.append(f"나이 {d['age']}세")
    if d.get("district"):
        parts.append(f"대전 {d['district']} 거주")
    parts.append(f"저소득 {yn(d.get('income_low'))}")
    parts.append(f"독거 {yn(d.get('living_alone'))}")
    parts.append(f"거동불편 {yn(d.get('mobility_issue'))}")
    parts.append(f"만성질환 {yn(d.get('chronic_illness'))}")
    return ", ".join(parts)


def _eligible_names(d: dict):
    eligible, _, _ = recommend(_profile_from(d))
    return [r.welfare.name for r in eligible]


# ── API 모델 ────────────────────────────────────────────────────────
class LoginIn(BaseModel):
    name: str


class ProfileIn(BaseModel):
    name: str
    age: int | None = None
    income_low: bool | None = None
    district: str | None = None
    living_alone: bool | None = None
    mobility_issue: bool | None = None
    chronic_illness: bool | None = None


class ChatIn(BaseModel):
    name: str
    question: str


@app.post("/login")
def login(inp: LoginIn):
    """이름으로 계정을 찾는다(없으면 신규)."""
    name = (inp.name or "").strip()
    if not name:
        return {"ok": False, "msg": "성함을 입력해 주세요."}
    store = _load_store()
    prof = store.get(name)
    return {"ok": True, "name": name, "profile": prof, "is_new": prof is None}


@app.post("/profile")
def save_profile(p: ProfileIn):
    """어르신 프로필 저장 + 받을 수 있는 복지 추천."""
    name = (p.name or "").strip()
    if not name:
        return {"ok": False, "msg": "성함을 입력해 주세요."}
    d = p.model_dump()
    d.pop("name")
    store = _load_store()
    store[name] = d
    _save_store(store)
    return {"ok": True, "eligible": _eligible_names(d), "summary": _profile_text(d)}


@app.post("/chat")
def chat(c: ChatIn):
    """어르신 프로필을 반영한 RAG 상담."""
    name = (c.name or "").strip()
    q = (c.question or "").strip()
    if not q:
        return {"answer": "궁금한 점을 입력해 주세요.", "sources": []}
    store = _load_store()
    d = store.get(name) or {}
    rag = _get_rag()
    if rag is None:
        return {"answer": "(상담 엔진이 준비되지 않았습니다 — 인덱스/Ollama 확인)",
                "sources": []}
    try:
        hits = rag.search(q, k=EVIDENCE_K)
        ctx = "\n\n".join(h[0] for h in hits)
        prof_line = _profile_text(d) if d else "정보 없음"
        elig = ", ".join(_eligible_names(d)) if d else ""
        prompt = (
            "당신은 대전 독거노인 복지 상담사입니다. 아래 [어르신 정보]와 [자료]만 "
            "근거로 한국어로 짧고 공손하게 답하세요. 어르신의 나이·소득·거주지 상황에 "
            "맞춰 안내하고, 자료에 없으면 '자료에 없습니다. 가까운 주민센터에 "
            "문의하세요'라고 답하세요.\n\n"
            f"[어르신 정보] {prof_line}\n"
            f"[규칙상 받을 수 있는 복지] {elig or '정보 부족'}\n\n"
            f"[자료]\n{ctx}\n\n[질문] {q}")
        import requests
        r = requests.post(rag.OLLAMA + "/api/generate",
                          json={"model": rag.CHAT_MODEL, "prompt": prompt,
                                "stream": False}, timeout=120)
        r.raise_for_status()
        answer = r.json().get("response", "").strip()
    except Exception as e:                 # noqa: BLE001
        return {"answer": f"(오류: {e}) Ollama 실행 여부를 확인하세요.", "sources": []}
    srcs = [{"source": s, "score": round(sim, 3)} for _, s, sim in hits
            if sim >= EVIDENCE_MIN_SIM]
    return {"answer": answer, "sources": srcs}


PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>대전노인복지 HUB</title>
<style>
  :root{--brand:#1d6fe0;--ink:#14213d;--line:#d9e2f0;--ok:#1f7a43;--okbg:#eaf2e9;--bot:#eef3fb;--muted:#5b6b86}
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,"Noto Sans KR",sans-serif;background:#f3f6fb;color:var(--ink)}
  header{background:var(--brand);color:#fff;padding:15px 18px}
  header h1{margin:0;font-size:19px;font-weight:800}
  header small{display:block;font-weight:500;font-size:12.5px;opacity:.9;margin-top:3px}
  main{max-width:720px;margin:0 auto;padding:16px}
  .card{background:#fff;border:1px solid var(--line);border-radius:16px;padding:18px;margin-bottom:14px}
  .step{font-size:13px;font-weight:800;color:var(--brand);margin-bottom:10px}
  label{display:block;font-size:13.5px;font-weight:700;margin:10px 0 5px}
  input,select{width:100%;border:1.5px solid var(--line);border-radius:10px;padding:11px 12px;font-size:16px;outline:none;background:#fff}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .btn{width:100%;margin-top:14px;border:0;border-radius:12px;background:var(--brand);color:#fff;font-size:16px;font-weight:800;padding:13px}
  .btn:disabled{background:#aebfd8}
  .hide{display:none}
  .elig{background:var(--okbg);border:1px solid #cadfc7;border-left:5px solid var(--ok);border-radius:12px;padding:12px 14px;margin:10px 0;font-size:14.5px}
  .elig b{color:var(--ok)}
  /* chat */
  #chat{display:flex;flex-direction:column;gap:10px;min-height:220px;max-height:50vh;overflow-y:auto;padding:4px}
  .row{display:flex;gap:8px;max-width:92%}
  .row.me{align-self:flex-end;flex-direction:row-reverse}
  .av{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;background:var(--bot);font-size:16px}
  .row.me .av{background:var(--brand)}
  .bub{padding:10px 13px;border-radius:15px;font-size:15.5px;line-height:1.5;white-space:pre-wrap}
  .bot .bub{background:var(--bot)} .me .bub{background:var(--brand);color:#fff}
  .src{margin-top:5px;font-size:11.5px;color:var(--muted)}
  .inrow{display:flex;gap:8px;margin-top:10px}
  #q{flex:1;border:1.5px solid var(--line);border-radius:20px;padding:11px 15px;font-size:16px;outline:none}
  #send{width:48px;border:0;border-radius:50%;background:var(--brand);color:#fff;font-size:20px}
  .note{font-size:12px;color:var(--muted);margin-top:10px}
  .link{color:var(--brand);font-weight:700;cursor:pointer;font-size:13px}
</style></head><body>
<header><h1>🏠 대전노인복지 HUB</h1>
<small>내 정보를 넣으면, 나에게 맞는 복지를 알려주고 상담해 드립니다</small></header>
<main>

  <!-- STEP 1: 로그인 -->
  <div class="card" id="c1">
    <div class="step">1단계 · 로그인</div>
    <label>성함 (계정)</label>
    <input id="name" placeholder="예: 홍길동" autocomplete="off">
    <button class="btn" id="loginBtn">시작하기</button>
    <div class="note">※ 프로토타입이라 성함만으로 구분합니다. 실제 서비스에서는 본인 인증·동의가 필요합니다.</div>
  </div>

  <!-- STEP 2: 프로필 -->
  <div class="card hide" id="c2">
    <div class="step">2단계 · 내 정보</div>
    <div class="grid">
      <div><label>나이 (만)</label><input id="age" type="number" min="0" placeholder="예: 72"></div>
      <div><label>거주 구</label><select id="district">
        <option value="">선택 안 함</option><option>동구</option><option>중구</option>
        <option>서구</option><option>유성구</option><option>대덕구</option></select></div>
      <div><label>저소득</label><select id="income_low">
        <option value="">모름</option><option value="true">예</option><option value="false">아니오</option></select></div>
      <div><label>독거</label><select id="living_alone">
        <option value="">모름</option><option value="true">예</option><option value="false">아니오</option></select></div>
      <div><label>거동 불편</label><select id="mobility_issue">
        <option value="">모름</option><option value="true">예</option><option value="false">아니오</option></select></div>
      <div><label>만성질환·기억력 저하</label><select id="chronic_illness">
        <option value="">모름</option><option value="true">예</option><option value="false">아니오</option></select></div>
    </div>
    <button class="btn" id="saveBtn">저장하고 상담 시작</button>
  </div>

  <!-- STEP 3: 상담 -->
  <div class="card hide" id="c3">
    <div class="step">3단계 · 맞춤 상담 <span class="link" id="editLink">[내 정보 수정]</span></div>
    <div id="eligBox"></div>
    <div id="chat"></div>
    <div class="inrow">
      <input id="q" placeholder="궁금한 복지를 물어보세요…" autocomplete="off">
      <button id="send">➤</button>
    </div>
    <div class="note">답변은 어르신 정보와 대전 공공문서 근거를 함께 반영합니다.</div>
  </div>

</main>
<script>
const $=id=>document.getElementById(id);
let USER=null;
function triv(v){return v===''?null:(v==='true');}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function show(step){ for(const i of [1,2,3]) $('c'+i).classList.toggle('hide', i!==step); }

async function post(url,body){
  const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  return r.json();
}

// STEP 1
$('loginBtn').onclick=async()=>{
  const name=$('name').value.trim(); if(!name){alert('성함을 입력해 주세요.');return;}
  const d=await post('/login',{name});
  if(!d.ok){alert(d.msg);return;}
  USER=d.name;
  if(d.profile){ // 기존 회원 → 프로필 채우고 바로 상담
    const p=d.profile;
    $('age').value=p.age??''; $('district').value=p.district||'';
    $('income_low').value=p.income_low===null?'':String(p.income_low);
    $('living_alone').value=p.living_alone===null?'':String(p.living_alone);
    $('mobility_issue').value=p.mobility_issue===null?'':String(p.mobility_issue);
    $('chronic_illness').value=p.chronic_illness===null?'':String(p.chronic_illness);
  }
  show(2);
};

// STEP 2
$('saveBtn').onclick=async()=>{
  const ageV=$('age').value.trim();
  const body={name:USER, age: ageV===''?null:parseInt(ageV,10),
    district:$('district').value||null, income_low:triv($('income_low').value),
    living_alone:triv($('living_alone').value), mobility_issue:triv($('mobility_issue').value),
    chronic_illness:triv($('chronic_illness').value)};
  const d=await post('/profile',body);
  if(!d.ok){alert(d.msg);return;}
  const names=d.eligible||[];
  $('eligBox').innerHTML='<div class="elig"><b>'+esc(USER)+'</b> 님, 받으실 수 있는 복지는 <b>'
    +(names.length?esc(names.join(', ')):'추가 정보 필요')+'</b> 입니다.<br>'
    +'<span style="font-size:12.5px;color:#5b6b86">'+esc(d.summary||'')+'</span></div>';
  $('chat').innerHTML='';
  addBot('안녕하세요, '+USER+' 님. 위 복지에 대해 더 궁금하신 점을 물어보세요. 😊');
  show(3);
};
$('editLink').onclick=()=>show(2);

// STEP 3 chat
function addRow(who){
  const r=document.createElement('div');r.className='row '+who;
  const a=document.createElement('div');a.className='av';a.textContent=who==='me'?'🙂':'🤖';
  const w=document.createElement('div');const b=document.createElement('div');b.className='bub';
  w.appendChild(b);r.append(a,w);$('chat').appendChild(r);$('chat').scrollTop=$('chat').scrollHeight;
  return {b,w};
}
function addBot(t){const {b}=addRow('bot');b.textContent=t;}
async function ask(){
  const t=$('q').value.trim(); if(!t)return; $('q').value='';
  addRow('me').b.textContent=t;
  const {b,w}=addRow('bot'); b.textContent='…';
  try{
    const d=await post('/chat',{name:USER,question:t});
    b.textContent=d.answer||'(답변 없음)';
    if(d.sources&&d.sources.length){
      const s=document.createElement('div');s.className='src';
      s.textContent='📄 근거: '+d.sources.map(x=>x.source+' ['+x.score+']').join(', ');
      w.appendChild(s);
    }
  }catch(e){b.textContent='오류가 발생했습니다.';}
  $('chat').scrollTop=$('chat').scrollHeight;
}
$('send').onclick=ask;
$('q').addEventListener('keydown',e=>{if(e.key==='Enter')ask();});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
