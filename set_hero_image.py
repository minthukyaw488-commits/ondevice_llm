"""
Set the home hero background image.

Usage:
    python set_hero_image.py /path/to/your/photo.png

Takes any image, crops it to the hero banner shape (wide + short) and saves it
to assets/hero_bg.png, which app.py picks up automatically (embedded locally as
a data URI - nothing is uploaded anywhere). Re-run any time to change the photo.
"""
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "assets" / "hero_bg.png"
# Wide, short banner. The CSS overlays a navy gradient for text legibility, so
# a fairly large image is fine; we cap the size to keep the page lightweight.
OUT_W, OUT_H = 1600, 520


def main():
    if len(sys.argv) != 2:
        print("Usage: python set_hero_image.py /path/to/your/photo.(png|jpg|jpeg|webp)")
        raise SystemExit(2)
    src = Path(sys.argv[1]).expanduser()
    if not src.exists():
        print(f"[!] File not found: {src}")
        raise SystemExit(1)
    try:
        from PIL import Image, ImageOps
    except ImportError:
        print("[!] Pillow가 필요합니다:  pip install pillow")
        raise SystemExit(1)

    img = Image.open(src)
    img = ImageOps.exif_transpose(img).convert("RGB")   # honour phone rotation
    # Crop-to-fill the banner shape (centre), then resize.
    fitted = ImageOps.fit(img, (OUT_W, OUT_H), method=Image.LANCZOS,
                          centering=(0.5, 0.42))         # bias slightly toward top
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    fitted.save(TARGET, format="PNG", optimize=True)
    kb = TARGET.stat().st_size / 1024
    print(f"✅ 배경 이미지를 저장했습니다: {TARGET}  ({OUT_W}x{OUT_H}, {kb:.0f} KB)")
    print("   앱을 새로고침하면 히어로 배경으로 표시됩니다.")


if __name__ == "__main__":
    main()
