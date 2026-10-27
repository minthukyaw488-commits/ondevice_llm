# 서버 배포 가이드 — 대전노인복지 HUB (Week 08 · AWS GPU 서버)

강의(Week 08)대로 팀 GPU 서버 한 대에 우리 프로젝트를 올린다.
앱은 **FastAPI 통합 앱(`app_personal.py`)** 을 자기 포트(예: 8001)로 띄운다.

> 전제: 팀장이 GPU 서버(g4dn.xlarge)를 만들고 **퍼블릭 IP · .pem 키 · 내 포트**를
> 공유했다. 포트는 보안 그룹에 8001–8003 이 열려 있어야 한다.

---

## 0. 내 포트 정하기
A=8001 · B=8002 · C=8003 중 **내 번호**를 기억. 아래 `8001` 을 내 번호로 바꿔 쓴다.

## 1. 서버 접속 (2주차와 동일)
```bash
chmod 400 ~/키파일.pem
ssh -i ~/키파일.pem ubuntu@서버IP
nvidia-smi          # Tesla T4 15360MiB 보이면 정상
```

## 2. 내 자리 만들고 코드 받기
```bash
mkdir -p ~/team/본인영문이름 && cd ~/team/본인영문이름
git clone https://github.com/minthukyaw488-commits/ondevice_llm.git
cd ondevice_llm
git checkout claude/elderly-welfare-voice-ai-8v6bii
```

## 3. 파이썬 패키지
```bash
sudo apt update && sudo apt install -y python3-pip
pip install fastapi uvicorn requests numpy --break-system-packages
```

## 4. Ollama 모델 (팀장이 한 번만 — 공용)
```bash
# Ollama 설치가 안 돼 있으면(팀장만):
curl -fsSL https://ollama.com/install.sh | sh
# 모델 받기(팀장만, 한 벌이면 됨):
ollama pull exaone3.5:2.4b      # 답변 모델
ollama pull bge-m3              # 임베딩(RAG 검색)
ollama list                     # 팀원도 이걸로 확인됨
```

## 5. ⚠ 데이터·인덱스 올리기 (git 에 없음 — 꼭 따로 업로드)
`data/welfare_docs/` 원본과 `rag_lecture/` 의 인덱스 파일은 **git 에 없다.**
내 PC(Mac)에서 서버로 올려야 RAG 근거가 나온다. 두 방법 중 하나:

**(A) 이미 만든 인덱스만 올리기 — 가장 빠름 (권장)**
Mac 터미널에서:
```bash
cd ~/ondevice_llm
scp -i ~/키파일.pem rag_lecture/vectors.npy rag_lecture/chunks.json rag_lecture/sources.json \
    ubuntu@서버IP:~/team/본인영문이름/ondevice_llm/rag_lecture/
```

**(B) 원본 문서를 올려 서버에서 인덱스 빌드**
```bash
# Mac → 서버로 원본 폴더 업로드
scp -i ~/키파일.pem -r data/welfare_docs \
    ubuntu@서버IP:~/team/본인영문이름/ondevice_llm/data/
# 서버에서 빌드(pymupdf 필요)
pip install pymupdf --break-system-packages
python3 rag_lecture/build_index.py     # "총 조각 수: ~1500" 나오면 성공
```
> FileZilla(3주차)로도 같은 폴더에 끌어다 올리면 된다(SFTP · 포트 22).

## 6. 앱 실행 (내 포트로)
```bash
cd ~/team/본인영문이름/ondevice_llm
python3 -m uvicorn app_personal:app --host 0.0.0.0 --port 8001
```
브라우저: **http://서버IP:8001**  → 로그인 → 내 정보 → 맞춤 상담

- 근거(RAG)가 나오려면 4·5단계가 끝나 있어야 한다.
- 안 나오면: `ollama list` 에 bge-m3·exaone3.5 가 있는지, 인덱스 파일이 있는지 확인.

## 7. 터미널 닫아도 계속 돌게 — systemd (응용 미션 A1 · 발표 전 필수)
```bash
# 서비스 파일 복사 (경로·포트·이름을 본인 것으로 수정 후)
sudo cp deploy/app_personal.service /etc/systemd/system/welfarehub-8001.service
sudo nano /etc/systemd/system/welfarehub-8001.service   # 경로/포트 확인
sudo systemctl daemon-reload
sudo systemctl enable --now welfarehub-8001
sudo systemctl status welfarehub-8001                   # active(running) 확인
```
이제 터미널·SSH 를 닫아도 서버가 계속 돈다. 로그: `journalctl -u welfarehub-8001 -f`

## 8. (선택) Apache 로 깔끔한 주소 — http://서버IP/내이름/
포트 번호 없이 보여주고 싶으면 Apache 리버스 프록시:
```bash
sudo apt install -y apache2
sudo a2enmod proxy proxy_http
# /etc/apache2/sites-available/000-default.conf 의 <VirtualHost> 안에 추가:
#   ProxyPass        /본인이름/ http://127.0.0.1:8001/
#   ProxyPassReverse /본인이름/ http://127.0.0.1:8001/
sudo systemctl reload apache2
```
> 경로 프록시는 앱이 상대경로를 쓸 때 가장 잘 맞는다. 단순하게는 그냥 `:8001` 로 써도 된다.

---

## 끝나기 전에 반드시 (강의 35p)
- 작업 끝나면 팀장이 **인스턴스 중지**(Terminate 금지 — 하면 전부 사라짐).
- 다음 주 켜면 **IP 가 바뀐다** → 5·6단계만 다시(코드·인덱스는 그대로 있음).

## 자주 나오는 문제 (강의 40p)
| 증상 | 원인·해결 |
|---|---|
| `Address already in use` | 내 포트를 남이 씀 → 내 번호 맞는지 확인 |
| 화면 비어 있음 | F12 → CORS 또는 IP 오타 (4주차) |
| 근거가 안 나옴 | Ollama 모델·인덱스 파일(5단계) 확인 |
| 답이 느림 | 셋이 동시 사용 → `nvidia-smi` 확인 |
| 어제 주소가 안 됨 | 서버 재시작으로 IP 바뀜 → 새 IP 사용 |

## 다음 단계 (팀 과제 T1)
지금은 프로필을 `data/profiles.json` 파일로 저장한다. 강의 MySQL 단계(STEP 06)에
맞춰 `users·profiles·chat_logs` 테이블로 옮기면 T1(우리 주제로 DB)이 완성된다.
필요하면 스키마와 연결 코드를 추가로 만들 수 있다.
