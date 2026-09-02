from .base import LocalManifestAdapter


class VoxSafeBenchAdapter(LocalManifestAdapter):
    benchmark_name = "VoxSafeBench"
    def setup_hint(self) -> str:
        return "Prepare VoxSafeBench locally and export acoustic-context-sensitive pairs to pairs.jsonl."
