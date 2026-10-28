from __future__ import annotations

import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def fetch(urls, dest: Path, retries: int = 6, timeout: float = 60.0) -> Path:
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last = None
    for url in ([urls] if isinstance(urls, str) else list(urls)):
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as r, open(part, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 16)
                part.replace(dest)
                return dest
            except urllib.error.HTTPError as e:
                part.unlink(missing_ok=True)
                last = e
                if e.code == 404:
                    break
            except (OSError, urllib.error.URLError) as e:
                part.unlink(missing_ok=True)
                last = e
            if attempt < retries - 1:
                wait = min(60, 5 * 2 ** attempt)
                print(f"download of {url} failed ({type(last).__name__}: {last}); retry {attempt + 1}/{retries - 1} "
                      f"in {wait} s", file=sys.stderr, flush=True)
                time.sleep(wait)
    raise last
