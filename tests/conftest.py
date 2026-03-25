import os

import pytest

VLM_PATH = os.getenv("VLM", "Qwen/Qwen3-VL-2B-Instruct")


@pytest.fixture(scope="session")
def vlm_path() -> str:
    return VLM_PATH
