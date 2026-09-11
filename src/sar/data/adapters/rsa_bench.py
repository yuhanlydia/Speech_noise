from .base import LocalManifestAdapter


class RSABenchAdapter(LocalManifestAdapter):
    benchmark_name = "RSA-Bench"
    def setup_hint(self) -> str:
        return "Prepare RSA-Bench locally and export same-waveform relevance records to pairs.jsonl; this repo never auto-downloads benchmark audio."
