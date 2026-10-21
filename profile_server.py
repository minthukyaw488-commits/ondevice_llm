"""
사람 맞춤형 복지 추천 — 웹 UI (FastAPI).

profile_match.py 의 규칙 기반 자격 매칭(src.welfare_rules)을 그대로 쓰고,
그 위에 '프로필 입력 폼' 웹페이지를 얹었다. 어르신 정보를 넣고 버튼을 누르면
받을 수 있는 복지를 자동 추천하고, 가능하면 실제 문서 근거(RAG)까지 보여준다.

설치·실행:
  pip install fastapi uvicorn
  # (근거 표시는 선택) rag_lecture 인덱스 + Ollama(bge-m3) 가 있으면 자동 사용
  python3 -m uvicorn profile_server:app --host 0.0.0.0 --port 8001

브라우저:  http://localhost:8001
"""
from __future__ import annotations
import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.welfare_rules import Profile, recommend

app = FastAPI(title="사람 맞춤형 복지 추천")

EVIDENCE_K = int(os.environ.get("RAG_EVIDENCE_K", "2"))
EVIDENCE_MIN_SIM = float(os.environ.get("RAG_EVIDENCE_MIN_SIM", "0.62"))

# rag_lecture/ask.py 를 한 번만 안전하게 로드(없으면 근거 생략).
_RAG = None
_RAG_TRIED = False


def _get_rag():
    global _RAG, _RAG_TRIED
    if _RAG_TRIED:
        return _RAG
    _RAG_TRIED = True
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    ask_path = os.path.join(here, "rag_lecture", "ask.py")
    try:
        spec = importlib.util.spec_from_file_location("rag_lecture_ask", ask_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _RAG = mod
    except Exception:                      # noqa: BLE001
        _RAG = None
    return _RAG


def _evidence(rag, name: str):
    """복지명으로 문서 근거를 검색해 유사도 게이트를 통과한 조각만 반환."""
    if rag is None:
        return []
    try:
        hits = rag.search(f"{name} 지원 대상 신청 자격", k=EVIDENCE_K)
    except Exception:                      # noqa: BLE001
        return []
    return [{"source": s, "score": round(sim, 3),
             "text": " ".join(c.split())[:90]}
            for c, s, sim in hits if sim >= EVIDENCE_MIN_SIM]


class ProfileIn(BaseModel):
    age: int | None = None
    income_low: bool | None = None
    district: str | None = None
    living_alone: bool | None = None
    mobility_issue: bool | None = None
    chronic_illness: bool | None = None


def _pack(results, with_evidence: bool):
    rag = _get_rag() if with_evidence else None
    out = []
    for r in results:
        item = {
            "name": r.welfare.name,
            "summary": r.welfare.summary,
            "reasons": r.reasons,
            "verify_note": r.welfare.verify_note,
        }
        if with_evidence:
            item["evidence"] = _evidence(rag, r.welfare.name)
        out.append(item)
    return out


@app.post("/recommend")
def do_recommend(p: ProfileIn):
    prof = Profile(age=p.age, income_low=p.income_low, district=p.district,
                   living_alone=p.living_alone, mobility_issue=p.mobility_issue,
                   chronic_illness=p.chronic_illness)
    eligible, maybe, not_elig = recommend(prof)
    return {
        "eligible": _pack(eligible, with_evidence=True),
        "maybe": _pack(maybe, with_evidence=False),
        "not": _pack(not_elig, with_evidence=False),
        "rag": _get_rag() is not None,
    }


PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>사람 맞춤형 복지 추천</title>
<style>
  :root{--brand:#1d6fe0;--ink:#14213d;--line:#d9e2f0;--ok:#1f7a43;--okbg:#eaf2e9;--muted:#5b6b86}
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,"Noto Sans KR",sans-serif;background:#f3f6fb;color:var(--ink)}
  header{background:var(--brand);color:#fff;padding:16px 18px}
  header h1{margin:0;font-size:20px;font-weight:800}
  header small{display:block;font-weight:500;font-size:13px;opacity:.9;margin-top:4px}
  main{max-width:760px;margin:0 auto;padding:18px}
  .card{background:#fff;border:1px solid var(--line);border-radius:16px;padding:20px;margin-bottom:16px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  label{display:block;font-size:14px;font-weight:700;margin-bottom:6px}
  input,select{width:100%;border:1.5px solid var(--line);border-radius:10px;padding:11px 12px;font-size:16px;outline:none;background:#fff}
  .btn{width:100%;margin-top:16px;border:0;border-radius:12px;background:var(--brand);color:#fff;font-size:17px;font-weight:800;padding:14px}
  .btn:disabled{background:#aebfd8}
  h2{font-size:16px;margin:18px 0 10px}
  .w{border:1px solid var(--line);border-left:5px solid var(--ok);border-radius:12px;padding:14px 16px;margin-bottom:12px;background:var(--okbg)}
  .w .nm{font-size:17px;font-weight:800}
  .w .sm{font-size:14px;color:#33415a;margin:4px 0 8px}
  .w .rs{font-size:13.5px;color:#2b3a52;margin:2px 0}
  .w .vf{font-size:12.5px;color:var(--muted);margin-top:6px}
  .ev{margin-top:8px;padding:10px;background:#fff;border:1px dashed var(--line);border-radius:8px}
  .ev .t{font-size:12.5px;color:#33415a;margin:3px 0}
  .ev .s{color:var(--brand);font-weight:700}
  .maybe,.no{border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin-bottom:8px;font-size:14px}
  .maybe{border-left:5px solid #e0a11d;background:#fdf7ea}
  .no{border-left:5px solid #c0c7d4;background:#f6f8fb;color:#5b6b86}
  .note{font-size:12.5px;color:var(--muted);margin-top:14px}
</style></head><body>
<header><h1>🧓 사람 맞춤형 복지 추천</h1>
<small>어르신 정보를 넣으면 받을 수 있는 복지를 자동으로 찾아 드립니다 (규칙 매칭 + 문서 근거 RAG)</small></header>
<main>
  <div class="card">
    <div class="grid">
      <div><label>나이 (만)</label><input id="age" type="number" min="0" placeholder="예: 72"></div>
      <div><label>거주 구</label><select id="district">
        <option value="">선택 안 함</option><option>동구</option><option>중구</option>
        <option>서구</option><option>유성구</option><option>대덕구</option></select></div>
      <div><label>저소득 (기초수급·차상위 등)</label><select id="income_low">
        <option value="">모름</option><option value="true">예</option><option value="false">아니오</option></select></div>
      <div><label>독거 (혼자 거주)</label><select id="living_alone">
        <option value="">모름</option><option value="true">예</option><option value="false">아니오</option></select></div>
      <div><label>거동 불편</label><select id="mobility_issue">
        <option value="">모름</option><option value="true">예</option><option value="false">아니오</option></select></div>
      <div><label>만성질환·기억력 저하</label><select id="chronic_illness">
        <option value="">모름</option><option value="true">예</option><option value="false">아니오</option></select></div>
    </div>
    <button class="btn" id="go">받을 수 있는 복지 찾기</button>
  </div>
  <div id="out"></div>
</main>
<script>
const $=id=>document.getElementById(id);
function triv(v){return v===''?null:(v==='true');}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
async function go(){
  $('go').disabled=true;$('out').innerHTML='<p class="note">찾는 중…</p>';
  const ageV=$('age').value.trim();
  const body={
    age: ageV===''?null:parseInt(ageV,10),
    district: $('district').value||null,
    income_low: triv($('income_low').value),
    living_alone: triv($('living_alone').value),
    mobility_issue: triv($('mobility_issue').value),
    chronic_illness: triv($('chronic_illness').value),
  };
  try{
    const r=await fetch('/recommend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    render(await r.json());
  }catch(e){$('out').innerHTML='<p class="note">오류가 발생했습니다.</p>';}
  $('go').disabled=false;
}
function render(d){
  let h='';
  h+='<h2>✅ 받을 수 있는 복지 ('+d.eligible.length+')</h2>';
  if(!d.eligible.length) h+='<div class="no">조건에 맞는 복지가 없습니다. 아래 ‘확인 필요’를 참고하세요.</div>';
  for(const w of d.eligible){
    h+='<div class="w"><div class="nm">'+esc(w.name)+'</div><div class="sm">'+esc(w.summary)+'</div>';
    for(const rs of w.reasons) h+='<div class="rs">· '+esc(rs)+'</div>';
    h+='<div class="vf">※ '+esc(w.verify_note)+'</div>';
    if(w.evidence&&w.evidence.length){
      h+='<div class="ev"><b>📄 문서 근거(RAG)</b>';
      for(const e of w.evidence) h+='<div class="t"><span class="s">['+e.score+']</span> ('+esc(e.source)+') '+esc(e.text)+'…</div>';
      h+='</div>';
    } else if(d.rag){
      h+='<div class="ev"><span class="t">📄 관련도 높은 근거 없음 (데이터 보강 과제)</span></div>';
    }
    h+='</div>';
  }
  if(d.maybe.length){
    h+='<h2>❓ 확인하면 받을 수 있음</h2>';
    for(const w of d.maybe){
      const need=w.reasons.filter(x=>x.indexOf('정보 필요')>=0);
      h+='<div class="maybe"><b>'+esc(w.name)+'</b> — '+esc(need.join(', ')||'추가 확인')+'</div>';
    }
  }
  if(d.not.length){
    h+='<h2>✖ 현재 조건상 대상 아님</h2>';
    for(const w of d.not) h+='<div class="no"><b>'+esc(w.name)+'</b> — '+esc(w.reasons[0]||'')+'</div>';
  }
  h+='<p class="note">→ 정확한 신청·금액은 주민센터 또는 복지 상담(RAG)으로 확인하세요. '+(d.rag?'(RAG 근거 ON)':'(RAG 근거 OFF — 인덱스/Ollama 없음)')+'</p>';
  $('out').innerHTML=h;
}
$('go').onclick=go;
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
