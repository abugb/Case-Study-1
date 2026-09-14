import sys
from unittest.mock import MagicMock, patch
import pytest

# Mock transformers.pipeline to prevent any network calls or 7B model downloads during unit tests
if "transformers" in sys.modules:
    sys.modules["transformers"].pipeline = MagicMock(return_value=None)
else:
    mock_tf = MagicMock()
    mock_tf.pipeline = MagicMock(return_value=None)
    sys.modules["transformers"] = mock_tf


@pytest.fixture(autouse=True)
def _no_log_metrics():
    """Prevent unit tests from writing to inference_metrics.log.

    In E2E CI environments where gradio is not installed, the patch
    target (app module) cannot be imported — yield a no-op instead.
    """
    try:
        with patch("app.log_inference_metrics") as mock_log:
            yield mock_log
    except ModuleNotFoundError:
        yield None
