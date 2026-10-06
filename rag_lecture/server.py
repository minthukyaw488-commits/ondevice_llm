"""
교수님 오늘 과제 — 팀 자료 RAG 답변을 '웹페이지에서 주고받기' (FastAPI).

ask.py 의 RAG(임베딩·검색·"자료만 근거로" 답변, 로컬 Ollama)를 그대로 쓰고,
그 위에 웹 서버를 얹었습니다. 브라우저에서 질문을 보내면 서버가 RAG로 답해
JSON으로 돌려주고, 같은 페이지에 표시합니다.

설치:
  pip install fastapi uvicorn            # (필요시 --break-system-packages)
  ollama pull bge-m3 && ollama pull exaone3.5:7.8b
  python rag_lecture/build_index.py      # 벡터가 없으면 먼저 1회

실행 (교수님 명령 그대로, rag_lecture 폴더에서):
  python3 -m uvicorn server:app --host 0.0.0.0 --port 8000

브라우저:  http://localhost:8000   (같은 WiFi의 다른 기기는 http://서버IP:8000)
"""
import ask  # rag_lecture/ask.py — search/ask + 벡터 로드 (import 시 1회)

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="대전 독거노인 복지 RAG")


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask_endpoint(q: Question):
    """질문 하나를 받아 RAG로 답하고, 답변과 근거(출처·유사도)를 돌려준다."""
    text = (q.question or "").strip()
    if not text:
        return {"answer": "질문을 입력해 주세요.", "sources": []}
    try:
        answer, hits = ask.ask(text)
    except Exception as e:                       # Ollama 미실행 등
        return {"answer": f"(오류: {e}) Ollama 실행 여부를 확인하세요.", "sources": []}
    return {
        "answer": answer.strip(),
        "sources": [
            {"source": src, "score": round(score, 3), "text": chunk[:80].strip()}
            for chunk, src, score in hits
        ],
    }


# --- 웹페이지 (어르신용, 한국어·큰 글씨) -----------------------------------
PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>대전 복지 상담</title>
<style>
  :root{--brand:#1d6fe0;--bot:#eef3fb;--me:#1d6fe0;--ink:#1a2b4a;--line:#d9e2f0}
  *{box-sizing:border-box} body{margin:0;font-family:-apple-system,"Noto Sans KR",sans-serif;
    background:#f3f6fb;color:var(--ink);display:flex;flex-direction:column;height:100vh}
  header{background:var(--brand);color:#fff;padding:14px 16px;font-size:19px;font-weight:800}
  header small{display:block;font-weight:500;font-size:12.5px;opacity:.9;margin-top:3px}
  main{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}
  .row{display:flex;gap:9px;align-items:flex-end;max-width:92%}
  .row.me{align-self:flex-end;flex-direction:row-reverse}
  .av{width:34px;height:34px;border-radius:50%;display:grid;place-items:center;background:var(--bot);font-size:18px}
  .row.me .av{background:var(--me)}
  .bub{padding:11px 14px;border-radius:16px;font-size:16.5px;line-height:1.55;white-space:pre-wrap}
  .bot .bub{background:var(--bot)} .me .bub{background:var(--me);color:#fff}
  .src{margin-top:6px;font-size:12px;color:#5b6b86}
  footer{display:flex;gap:9px;padding:10px 12px;background:#fff;border-top:1px solid var(--line)}
  #q{flex:1;border:1.5px solid var(--line);border-radius:22px;padding:12px 16px;font-size:16.5px;outline:none}
  #send{width:52px;border:0;border-radius:50%;background:var(--brand);color:#fff;font-size:22px}
  #send:disabled{background:#aebfd8}
</style></head><body>
<header>🏠 대전 독거노인 복지 상담 <small>질문하시면 대전 복지 자료에 근거해 답해 드립니다 (RAG · 로컬 LLM)</small></header>
<main id="chat"></main>
<footer>
  <input id="q" placeholder="복지 서비스를 물어보세요…" autocomplete="off">
  <button id="send">➤</button>
</footer>
<script>
const chat=document.getElementById('chat'),q=document.getElementById('q'),send=document.getElementById('send');
function add(who,text,srcs){
  const r=document.createElement('div');r.className='row '+who;
  const a=document.createElement('div');a.className='av';a.textContent=who==='me'?'🙂':'🤖';
  const w=document.createElement('div');const b=document.createElement('div');b.className='bub';b.textContent=text;w.appendChild(b);
  if(srcs&&srcs.length){const s=document.createElement('div');s.className='src';s.textContent='📄 근거: '+srcs.map(x=>x.source).join(', ');w.appendChild(s);}
  r.append(a,w);chat.appendChild(r);chat.scrollTop=chat.scrollHeight;return b;
}
add('bot','안녕하세요, 어르신. 궁금한 복지 서비스를 편하게 물어보세요. 😊');
async function ask(){
  const t=q.value.trim();if(!t)return;q.value='';send.disabled=true;
  add('me',t);const b=add('bot','…');
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:t})});
    const d=await r.json();b.textContent=d.answer;
    if(d.sources&&d.sources.length){const s=document.createElement('div');s.className='src';s.textContent='📄 근거: '+d.sources.map(x=>x.source).join(', ');b.parentNode.appendChild(s);}
  }catch(e){b.textContent='오류가 발생했습니다. 다시 시도해 주세요.';}
  send.disabled=false;q.focus();
}
send.onclick=ask;q.addEventListener('keydown',e=>{if(e.key==='Enter')ask();});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
