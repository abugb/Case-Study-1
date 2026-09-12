import sys
from unittest.mock import MagicMock

# Mock diffusers to prevent downloading multi-gigabyte models during unit testing
mock_diffusers = MagicMock()
mock_diffusers.ControlNetModel = MagicMock()
mock_diffusers.StableDiffusionControlNetPipeline = MagicMock()
mock_diffusers.StableVideoDiffusionPipeline = MagicMock()
mock_diffusers.UniPCMultistepScheduler = MagicMock()
sys.modules["diffusers"] = mock_diffusers

mock_utils = MagicMock()
mock_utils.export_to_video = MagicMock(return_value=None)
sys.modules["diffusers.utils"] = mock_utils

# Mock transformers
if "transformers" in sys.modules:
    sys.modules["transformers"].pipeline = MagicMock(return_value=None)
else:
    mock_tf = MagicMock()
    mock_tf.pipeline = MagicMock(return_value=None)
    sys.modules["transformers"] = mock_tf


