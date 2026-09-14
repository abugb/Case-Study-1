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
    """Prevent unit tests from writing to inference_metrics.log."""
    with patch("app.log_inference_metrics") as mock_log:
        yield mock_log
