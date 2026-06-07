from pathlib import Path


def test_runtime_dependencies_are_lightweight_and_local_only() -> None:
    pyproject = Path("pyproject.toml").read_text()

    assert '"pillow>=12.0.0"' in pyproject
    assert '"pillow-heif>=1.1.0"' in pyproject
    assert "requests" not in pyproject
    assert "httpx" not in pyproject
    assert "openai" not in pyproject
    assert "torch" not in pyproject
    assert "tensorflow" not in pyproject
    assert "transformers" not in pyproject
    assert "open-clip" not in pyproject


def test_source_has_no_network_or_cloud_runtime_imports() -> None:
    source_text = "\n".join(path.read_text() for path in Path("src").rglob("*.py"))

    forbidden_tokens = [
        "import requests",
        "import httpx",
        "import urllib",
        "import socket",
        "import subprocess",
        "from openai",
        "import openai",
        "http://",
        "https://",
    ]
    assert all(token not in source_text for token in forbidden_tokens)
