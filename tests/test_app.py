import base64
from io import BytesIO
from unittest.mock import MagicMock, patch
from PIL import Image
import gradio as gr
import pytest

from app import (
    extract_and_prepare_image,
    image_to_data_url,
    process_drawing,
    local_generate,
    demo,
)


class TestImageProcessing:
    def test_extract_empty_sketch(self):
        """Ensure empty sketchpad payloads return None."""
        assert extract_and_prepare_image(None) is None
        assert extract_and_prepare_image({}) is None
        assert extract_and_prepare_image({"composite": None}) is None

    def test_extract_alpha_compositing(self):
        """Ensure transparent strokes are alpha-blended onto a solid white background."""
        img = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
        img.putpixel((1, 0), (0, 0, 0, 255))

        payload = {"composite": img}
        result = extract_and_prepare_image(payload)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (2, 2)
        # Transparent pixel should become pure white (255, 255, 255)
        assert result.getpixel((0, 0)) == (255, 255, 255)
        # Opaque black pixel should remain black (0, 0, 0)
        assert result.getpixel((1, 0)) == (0, 0, 0)

    def test_image_to_data_url(self):
        """Ensure PIL images convert to valid base64 data URLs."""
        img = Image.new("RGB", (10, 10), color="blue")
        data_url = image_to_data_url(img)

        assert data_url.startswith("data:image/png;base64,")

        header, b64_str = data_url.split(",", 1)
        decoded = base64.b64decode(b64_str)
        reopened = Image.open(BytesIO(decoded))
        assert reopened.size == (10, 10)


class TestProcessDrawing:
    def test_empty_sketchpad(self):
        """Submitting an empty sketchpad should return 'Sketchpad is empty'."""
        assert process_drawing(None) == "Sketchpad is empty"
        assert process_drawing({}) == "Sketchpad is empty"
        assert process_drawing({"composite": None}) == "Sketchpad is empty"

    def test_remote_model_missing_hf_token(self, monkeypatch):
        """When HF_TOKEN is missing, return 'HF_TOKEN not found'."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        dummy_sketch = {"composite": Image.new("RGBA", (10, 10), (0, 0, 0, 255))}
        result = process_drawing(dummy_sketch, use_local_model=False)
        assert result == "HF_TOKEN not found"

    def test_remote_model_success(self, monkeypatch):
        """When HF_TOKEN is set and client returns a guess, process_drawing returns it."""
        monkeypatch.setenv("HF_TOKEN", "mock_token")
        dummy_sketch = {"composite": Image.new("RGBA", (10, 10), (0, 0, 0, 255))}

        mock_choice = MagicMock()
        mock_choice.message.content = "A Cat"
        mock_response = MagicMock(choices=[mock_choice])

        with patch("app.InferenceClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = process_drawing(dummy_sketch, use_local_model=False)
            assert result == "A Cat"
            mock_client.chat.completions.create.assert_called_once()

    def test_remote_model_failure(self, monkeypatch):
        """When InferenceClient raises an exception, return friendly error."""
        monkeypatch.setenv("HF_TOKEN", "mock_token")
        dummy_sketch = {"composite": Image.new("RGBA", (10, 10), (0, 0, 0, 255))}

        with patch("app.InferenceClient", side_effect=Exception("Connection timed out")):
            result = process_drawing(dummy_sketch, use_local_model=False)
            assert result == "Failed to connect to inference API"

    def test_local_model_dispatch(self):
        """When use_local_model is True, dispatch to local_generate."""
        dummy_sketch = {"composite": Image.new("RGBA", (10, 10), (0, 0, 0, 255))}

        with patch("app.local_generate", return_value="A Dog") as mock_local:
            result = process_drawing(dummy_sketch, use_local_model=True)
            assert result == "A Dog"
            mock_local.assert_called_once()


class TestLocalGenerate:
    def test_local_generate_no_output(self, monkeypatch):
        """When pipe returns an empty list, handle gracefully."""
        mock_pipe = MagicMock(return_value=[])
        monkeypatch.setattr("app.pipe", mock_pipe)
        result = local_generate([{"role": "user", "content": "test"}])
        assert result == "Model produced no output."

    def test_local_generate_exception(self, monkeypatch):
        """When pipe raises an exception, return error message."""
        mock_pipe = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
        monkeypatch.setattr("app.pipe", mock_pipe)
        result = local_generate([{"role": "user", "content": "test"}])
        assert "⚠️ Local Model Error: CUDA out of memory" in result


class TestGradioInterface:
    def test_interface_configuration(self):
        """Validate that the Gradio interface is properly configured."""
        assert isinstance(demo, gr.Interface)
        assert len(demo.input_components) == 2
        assert len(demo.output_components) == 1
