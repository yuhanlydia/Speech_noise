def test_core_modules_import_without_gpu_extras():
    import sar.config  # noqa: F401
    import sar.data.schema  # noqa: F401
    import sar.eval  # noqa: F401
    import sar.gpu_smoke  # noqa: F401
    import sar.methods.qacr  # noqa: F401
    import sar.models.qwen_hook  # noqa: F401
    import sar.models.qwen_omni  # noqa: F401
    import sar.train  # noqa: F401
