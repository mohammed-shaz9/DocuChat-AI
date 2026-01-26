from __future__ import annotations

import re
from typing import Dict, List


def is_valid_google_doc_url(url: str) -> bool:
    return isinstance(url, str) and "docs.google.com/document" in url and bool(re.search(r"/d/([a-zA-Z0-9_-]+)", url))


def require_json_fields(data: Dict[str, object], fields: List[str]) -> List[str]:
    missing = []
    for f in fields:
        v = data.get(f)
        if v is None or (isinstance(v, str) and not v.strip()):
            missing.append(f)
    return missing