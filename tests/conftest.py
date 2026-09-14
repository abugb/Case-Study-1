import sys
from unittest.mock import MagicMock

if "transformers" in sys.modules:
    sys.modules["transformers"].pipeline = MagicMock(return_value=None)
else:
    mock_tf = MagicMock()
    mock_tf.pipeline = MagicMock(return_value=None)
    sys.modules["transformers"] = mock_tf