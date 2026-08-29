"""
Desktop app launcher (tablet / PC / mini-PC).

Wraps the existing Streamlit portal in a chromeless, full-screen app window so
it looks and feels like a native application - not a browser tab. NOTHING in the
app code changes: this only starts Streamlit and opens it in Chrome/Edge/Brave
"app mode" (--app), which supports the microphone because localhost is a secure
context. On-device LLM, RAG, voice and the worker are all unchanged.

Run (use the interpreter that has the models, e.g. rag_env):
    python desktop_app.py

Quit: close the app window, then press Ctrl+C in this terminal.
For an unattended kiosk (mini-PC at a home), swap --app for --kiosk below.
"""
import atexit
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request

PORT = 8501
URL = f"http://localhost:{PORT}"
PROFILE = os.path.expanduser("~/.welfare_app_profile")  # remembers mic permission


def wait_ready(timeout: int = 240) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(f"{URL}/_stcore/health", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def find_browser():
    """Return (kind, target): ('mac', app name) | ('exe', path) | (None, None)."""
    system = platform.system()
    if system == "Darwin":
        for app in ["Google Chrome", "Microsoft Edge", "Brave Browser", "Chromium"]:
            if os.path.exists(f"/Applications/{app}.app"):
                return "mac", app
    if system == "Windows":
        import glob
        pats = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        ]
        for p in pats:
            if glob.glob(p):
                return "exe", glob.glob(p)[0]
    for exe in ["google-chrome", "chromium", "chromium-browser",
                "microsoft-edge", "brave-browser"]:
        path = shutil.which(exe)
        if path:
            return "exe", path
    return None, None


def launch_window():
    kind, target = find_browser()
    flags = [
        f"--app={URL}",                       # chromeless app window (use --kiosk for locked kiosk)
        "--start-maximized",
        f"--user-data-dir={PROFILE}",
        "--no-first-run", "--no-default-browser-check",
    ]
    if kind == "mac":
        subprocess.run(["open", "-na", target, "--args", *flags])
        return True
    if kind == "exe":
        subprocess.Popen([target, *flags])
        return True
    print(f"\n[!] Chrome/Edge를 찾지 못했습니다. 브라우저에서 직접 열어 주세요: {URL}\n")
    return False


def main():
    print("복지 도우미 앱을 시작합니다… (모델 로딩까지 잠시 걸립니다)")
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.headless=true", f"--server.port={PORT}",
         "--browser.gatherUsageStats=false"],
        cwd=os.path.dirname(os.path.abspath(__file__)))
    atexit.register(lambda: server.terminate())

    if not wait_ready():
        print("Streamlit 서버를 시작하지 못했습니다.")
        server.terminate()
        return

    launch_window()
    print(f"\n✅ 앱이 실행되었습니다. 창을 닫은 뒤 이 터미널에서 Ctrl+C 로 종료하세요.\n   ({URL})")
    try:
        server.wait()
    except KeyboardInterrupt:
        print("\n종료합니다.")
        server.terminate()


if __name__ == "__main__":
    main()
