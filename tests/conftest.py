import sys
from unittest.mock import Mock

import pytest
from PIL import Image


@pytest.fixture(scope="session")
def app_module():
    # Load the UI without importing ML runtimes or downloading weights.
    modules = {
        "torch": Mock(),
        "transformers": Mock(pipeline=Mock(return_value=Mock(return_value=[]))),
    }
    with pytest.MonkeyPatch.context() as setup:
        for name, module in modules.items():
            setup.setitem(sys.modules, name, module)
        setup.setenv("HF_TOKEN", "unit-test-token")
        import app
    return app


@pytest.fixture
def app(app_module, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "unit-test-token")
    return app_module


@pytest.fixture
def drawing():
    return Image.new("RGB", (8, 8), "white")


@pytest.fixture
def remote_client(app, monkeypatch):
    client = Mock()
    monkeypatch.setattr(app, "InferenceClient", Mock(return_value=client))
    return client
