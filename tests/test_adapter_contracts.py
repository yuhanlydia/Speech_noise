from pathlib import Path
import pytest

from sar.data.adapters.mmsu import MMSUAdapter
from sar.data.adapters.rsa_bench import RSABenchAdapter
from sar.data.adapters.sh_bench import SHBenchAdapter
from sar.data.adapters.voxsafebench import VoxSafeBenchAdapter


@pytest.mark.parametrize("adapter_cls,name", [
    (RSABenchAdapter, "RSA-Bench"),
    (MMSUAdapter, "MMSU"),
    (SHBenchAdapter, "SH-Bench"),
    (VoxSafeBenchAdapter, "VoxSafeBench"),
])
def test_missing_benchmark_root_is_actionable(tmp_path: Path, adapter_cls, name):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match=name):
        list(adapter_cls().iter_records(missing))
