"""
Data inspection report for data/welfare_docs/.

Prints, for every file that gets indexed, what is actually inside it — so you
can see at a glance whether a table carries off-domain rows (동물병원, 자동차 등),
English/URL noise, or unexpected columns that let out-of-scope questions match.

Run:  python data_report.py
(no models needed; pure file reading)
"""
from __future__ import annotations
import csv
import re
from pathlib import Path

from src import config

URL_RE = re.compile(r"https?://|www\.|\.or\.kr|\.go\.kr|\.com|\.net")
LATIN_RE = re.compile(r"[A-Za-z]{2,}")
# Words that suggest a row is NOT elderly-welfare content (spurious matches).
OFF_DOMAIN = ["동물병원", "자동차", "판매점", "부동산", "주식", "호텔", "항공",
              "여행사", "카페", "커피", "넷플릭스", "세탁기", "노트북", "스마트폰"]
# Words that mark a row as on-topic elderly welfare.
WELFARE = ["노인", "어르신", "독거", "복지", "돌봄", "치매", "요양", "경로",
           "급식", "재가", "연금", "바우처", "정신건강", "보건소", "복지관",
           "생활지원", "안심", "일자리", "돌봄", "방문건강"]


def _rows_from_csv(path: Path):
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            with path.open(encoding=enc, newline="") as f:
                rows = list(csv.reader(f))
            return rows, enc
        except (UnicodeDecodeError, LookupError):
            continue
    return None, None


def _rows_from_excel(path: Path):
    try:
        import pandas as pd
    except Exception:
        return None, "pandas 미설치"
    for engine in (None, "openpyxl", "xlrd"):
        try:
            frames = pd.read_excel(path, sheet_name=None, dtype=str, engine=engine)
            rows = []
            for name, df in frames.items():
                df = df.fillna("")
                rows.append([str(c) for c in df.columns])
                for _, r in df.iterrows():
                    rows.append([str(v) for v in r])
            return rows, f"pandas({len(frames)} sheet)"
        except Exception:
            continue
    return None, "엑셀 읽기 실패"


def analyse_table(name: str, rows):
    if not rows:
        print(f"  (빈 표 또는 읽기 실패)")
        return
    header, data = rows[0], rows[1:]
    print(f"  열({len(header)}): {', '.join(str(h) for h in header)[:100]}")
    print(f"  행 수: {len(data)}")
    joined = ["  ".join(str(c) for c in r) for r in data]
    off = [j for j in joined if any(w in j for w in OFF_DOMAIN)]
    onw = [j for j in joined if any(w in j for w in WELFARE)]
    urls = [j for j in joined if URL_RE.search(j)]
    eng = [j for j in joined if LATIN_RE.search(j)]
    print(f"  복지 관련 행: {len(onw)}/{len(data)}   "
          f"의심(off-domain) 행: {len(off)}   URL 포함: {len(urls)}   영어 포함: {len(eng)}")
    if off:
        print("  ⚠️ off-domain 의심 예시:")
        for j in off[:3]:
            print(f"     - {j[:90]}")
    print("  샘플 행:")
    for j in joined[:3]:
        print(f"     · {j[:90]}")


def analyse_text(path: Path, text: str):
    print(f"  글자 수: {len(text)}")
    urls = URL_RE.findall(text)
    eng = LATIN_RE.findall(text)
    print(f"  URL 등장: {len(urls)}   영어 토큰: {len(eng)}")
    print("  앞부분:")
    snippet = " ".join(text.split())[:180]
    print(f"     {snippet}")


def main():
    docs = config.WELFARE_DOCS_DIR
    files = sorted(p for p in docs.glob("*") if p.is_file())
    print("=" * 70)
    print(f"데이터 폴더: {docs}   (파일 {len(files)}개)")
    print("=" * 70)
    for path in files:
        suf = path.suffix.lower()
        print(f"\n📄 {path.name}   [{suf}]   {path.stat().st_size//1024} KB")
        print("-" * 70)
        if suf == ".csv":
            rows, enc = _rows_from_csv(path)
            print(f"  인코딩: {enc}")
            analyse_table(path.name, rows)
        elif suf in (".xls", ".xlsx"):
            rows, info = _rows_from_excel(path)
            print(f"  읽기: {info}")
            analyse_table(path.name, rows)
        elif suf in (".md", ".txt"):
            analyse_text(path, path.read_text(encoding="utf-8", errors="ignore"))
        elif suf == ".pdf":
            print("  (PDF — 표 분석 대상 아님. 정책 원문으로 색인됨)")
        else:
            print("  (미지원 형식 — 색인 안 됨)")
    print("\n" + "=" * 70)
    print("해석 가이드:")
    print("  · '의심(off-domain) 행'이 많으면 → 그 표가 out-of-scope 오답의 원인일 수 있음")
    print("  · 'URL/영어 포함'이 많으면 → 답변에 영어가 새는 원인 (한국어 준수 하락)")
    print("=" * 70)


if __name__ == "__main__":
    main()
