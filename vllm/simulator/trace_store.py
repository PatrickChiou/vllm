import hashlib
import json
from array import array
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class TraceEntry:
    """Recorded response for one prompt.

    Either the exact `output_token_ids` are known, or only the number of
    generated tokens (`output_len`), in which case a filler token is replayed.
    """
    output_token_ids: Optional[Tuple[int, ...]]
    output_len: int


def prompt_key(prompt_token_ids: Sequence[int]) -> bytes:
    """Order-sensitive digest of a prompt, stable across processes."""
    return hashlib.sha1(array("q", prompt_token_ids).tobytes()).digest()


def _parse_entry(raw: Dict[str, Any], where: str) -> Tuple[bytes, TraceEntry]:
    if "prompt_token_ids" not in raw:
        raise ValueError(f"{where}: missing 'prompt_token_ids'.")
    prompt = [int(t) for t in raw["prompt_token_ids"]]
    if "output_token_ids" in raw:
        output = tuple(int(t) for t in raw["output_token_ids"])
        entry = TraceEntry(output_token_ids=output, output_len=len(output))
    elif "output_len" in raw:
        output_len = int(raw["output_len"])
        if output_len < 0:
            raise ValueError(f"{where}: 'output_len' must be non-negative.")
        entry = TraceEntry(output_token_ids=None, output_len=output_len)
    else:
        raise ValueError(
            f"{where}: needs 'output_token_ids' or 'output_len'.")
    return prompt_key(prompt), entry


class TraceStore:
    """Maps prompt token ids to recorded responses.

    The same prompt may occur several times with different responses (e.g.
    a common first turn such as "hi"); lookups cycle through them in trace
    order so that repeated prompts replay every recorded response.
    """

    def __init__(self, entries: Iterable[Dict[str, Any]] = ()) -> None:
        self._entries: Dict[bytes, List[TraceEntry]] = {}
        self._next_index: Dict[bytes, int] = {}
        self._num_entries = 0
        for i, raw in enumerate(entries):
            key, entry = _parse_entry(raw, f"trace entry {i}")
            self._entries.setdefault(key, []).append(entry)
            self._num_entries += 1

    @classmethod
    def from_file(cls, path: str) -> "TraceStore":
        """Loads a JSONL file, or a JSON file holding a list of entries."""
        with open(path, encoding="utf-8") as f:
            text = f.read()
        stripped = text.lstrip()
        if stripped.startswith("["):
            entries = json.loads(stripped)
        else:
            entries = [
                json.loads(line) for line in text.splitlines()
                if line.strip()
            ]
        store = cls(entries)
        logger.info("Loaded %d trace entries (%d unique prompts) from %s",
                    len(store), store.num_unique_prompts, path)
        return store

    def __len__(self) -> int:
        return self._num_entries

    @property
    def num_unique_prompts(self) -> int:
        return len(self._entries)

    def __contains__(self, prompt_token_ids: Sequence[int]) -> bool:
        return prompt_key(prompt_token_ids) in self._entries

    def lookup(self, prompt_token_ids: Sequence[int]) -> Optional[TraceEntry]:
        key = prompt_key(prompt_token_ids)
        candidates = self._entries.get(key)
        if not candidates:
            return None
        index = self._next_index.get(key, 0)
        self._next_index[key] = index + 1
        return candidates[index % len(candidates)]
