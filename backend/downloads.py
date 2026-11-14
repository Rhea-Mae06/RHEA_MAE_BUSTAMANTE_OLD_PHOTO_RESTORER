from __future__ import annotations

import shutil
from pathlib import Path
from urllib.request import Request, urlopen


def ensure_file(path: Path, url: str, min_bytes: int = 1024, timeout: int = 600) -> bool:
    if path.exists() and path.stat().st_size >= min_bytes:
        return True
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        request = Request(url, headers={"User-Agent": "OldPhotoRestorer/1.0"})
        with urlopen(request, timeout=timeout) as response, open(tmp, "wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        tmp.replace(path)
        return True
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False
