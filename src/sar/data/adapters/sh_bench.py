from .base import LocalManifestAdapter


class SHBenchAdapter(LocalManifestAdapter):
    benchmark_name = "SH-Bench"
    def setup_hint(self) -> str:
        return "Prepare SH-Bench locally and export selective/general paired records to pairs.jsonl."
