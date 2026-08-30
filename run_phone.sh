#!/usr/bin/env bash
# Launch the welfare portal over HTTPS on the LAN, so a phone can use the
# microphone (browsers only allow getUserMedia in a secure context).
#
# Everything stays on your local network - no cloud, no tunnel. The phone just
# needs to trust the local certificate ONCE (see the printed steps).
#
# Usage (from the env that has the models, e.g. `conda activate welfare`):
#   ./run_phone.sh
#
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8503}"
CERT_DIR="certs"
CERT="$CERT_DIR/cert.pem"
KEY="$CERT_DIR/key.pem"
mkdir -p "$CERT_DIR"

# --- find this Mac's LAN IP (Wi-Fi first, then Ethernet) --------------------
IP=""
for IF in en0 en1 en2; do
  IP="$(ipconfig getifaddr "$IF" 2>/dev/null || true)"
  [ -n "$IP" ] && break
done
if [ -z "$IP" ]; then
  IP="$(ifconfig 2>/dev/null | awk '/inet /{print $2}' | grep -v '^127\.' | head -1 || true)"
fi
[ -z "$IP" ] && { echo "[!] LAN IP를 찾지 못했습니다. Wi-Fi 연결을 확인하세요."; exit 1; }

# --- make a locally-trusted certificate ------------------------------------
CAROOT_MSG=""
if [ ! -f "$CERT" ] || [ ! -f "$KEY" ]; then
  if command -v mkcert >/dev/null 2>&1; then
    echo "→ mkcert로 인증서를 만듭니다 (IP: $IP)…"
    mkcert -install >/dev/null 2>&1 || true
    mkcert -cert-file "$CERT" -key-file "$KEY" "$IP" localhost 127.0.0.1 >/dev/null
    CAROOT_MSG="휴대폰 신뢰용 루트 인증서:  $(mkcert -CAROOT)/rootCA.pem"
  else
    echo "→ openssl로 자체 서명 인증서를 만듭니다 (IP: $IP)…"
    openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
      -keyout "$KEY" -out "$CERT" \
      -subj "/CN=welfare-hub" \
      -addext "subjectAltName=IP:${IP},DNS:localhost,IP:127.0.0.1" >/dev/null 2>&1
    CAROOT_MSG="휴대폰 신뢰용 인증서:  $(pwd)/$CERT   (더 쉬운 방법: brew install mkcert)"
  fi
else
  command -v mkcert >/dev/null 2>&1 && CAROOT_MSG="휴대폰 신뢰용 루트 인증서:  $(mkcert -CAROOT)/rootCA.pem"
fi

cat <<EOF

──────────────────────────────────────────────────────────────
 휴대폰에서 사용 (같은 Wi-Fi):   https://$IP:$PORT
 (음성 마이크는 https 에서만 동작합니다)

 휴대폰이 인증서를 한 번만 신뢰하도록 설정하세요:
   $CAROOT_MSG
   - iPhone: 위 파일을 휴대폰으로 보내 설치 →
             설정 > 일반 > VPN 및 기기 관리 에서 프로파일 설치 →
             설정 > 일반 > 정보 > 인증서 신뢰 설정 에서 '전체 신뢰' 켜기
   - Android: 설정 > 보안 > 인증서 설치(CA 인증서) 로 설치
 그 후 홈 화면에 추가하면 앱처럼 전체화면으로 열립니다.
──────────────────────────────────────────────────────────────

EOF

# WORKER_PYTHON defaults to this interpreter; make sure you ran this from the
# env that has the models (torch/faster-whisper/chromadb).
exec python -m streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port "$PORT" \
  --server.sslCertFile "$CERT" \
  --server.sslKeyFile "$KEY" \
  --browser.gatherUsageStats false
