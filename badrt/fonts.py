"""System fonts for the HUD and the arena textures (Chinese + a condensed sports face for numbers)."""
from __future__ import annotations

import os
from functools import lru_cache

from PIL import ImageFont

_WIN = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
CJK = {False: ["msyh.ttc", "msyh.ttf", "/System/Library/Fonts/PingFang.ttc",
               "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "simhei.ttf"],
       True: ["msyhbd.ttc", "msyhbd.ttf", "/System/Library/Fonts/PingFang.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "simhei.ttf"]}
LATIN = {False: ["bahnschrift.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"],
         True: ["bahnschrift.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"]}


def _find(names):
    for n in names:
        for p in (n, os.path.join(_WIN, n)):
            if os.path.isfile(p):
                return p
    return None


@lru_cache(maxsize=256)
def font(size, bold=False, cjk=False):
    """cjk=True: a face with Chinese glyphs (it has Latin too); False: the display face for
    numbers and Latin wordmarks."""
    size = int(round(size))
    path = _find(CJK[bold] if cjk else LATIN[bold])
    if path is None and not cjk:
        path = _find(CJK[bold])
    if path is None:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)        # bundled with Pillow
        except OSError:
            return ImageFont.load_default()
    f = ImageFont.truetype(path, size)
    if os.path.basename(path).lower() == "bahnschrift.ttf":
        try:
            f.set_variation_by_name("Bold SemiCondensed" if bold else "SemiCondensed")
        except Exception:
            try:
                f.set_variation_by_name("Bold" if bold else "Regular")
            except Exception:
                pass
    return f


def has_cjk():
    return _find(CJK[False]) is not None
