from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from sar.data.relevance_pairs import load_jsonl_records
from sar.data.schema import RelevancePairRecord


class LocalManifestAdapter(ABC):
    benchmark_name: str
    manifest_name: str = "pairs.jsonl"

    @abstractmethod
    def setup_hint(self) -> str: ...

    def iter_records(self, root: Path) -> Iterator[RelevancePairRecord]:
        root = Path(root)
        manifest = root / self.manifest_name
        if not root.exists() or not manifest.exists():
            raise FileNotFoundError(
                f"{self.benchmark_name}: expected a local exported manifest at {manifest}. {self.setup_hint()}"
            )
        yield from load_jsonl_records(manifest)
