"""Export study measurements and publication-friendly matplotlib figures."""
import json
from pathlib import Path
from typing import List


def read_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]