from .base import LocalManifestAdapter


class MMSUAdapter(LocalManifestAdapter):
    benchmark_name = "MMSU"
    def setup_hint(self) -> str:
        return "Prepare MMSU locally and export evaluation records to pairs.jsonl; use its acoustic-evidence tasks for the use side."
