import os
from unittest.mock import MagicMock, patch
from PIL import Image
import gradio as gr
import pytest

from app import (
    extract_and_prepare_image,
    prepare_controlnet_conditioning,
    process_drawing,
    local_generate_pipeline,
    demo,
)


class TestImageProcessing:
    def test_extract_empty_sketch(self):
        """Ensure empty sketchpad payloads and blank white canvases return None."""
        assert extract_and_prepare_image(None) is None
        assert extract_and_prepare_image({}) is None
        assert extract_and_prepare_image({"composite": None}) is None

        # Blank white canvas should also be detected as empty
        blank_canvas = Image.new("RGBA", (10, 10), (255, 255, 255, 255))
        assert extract_and_prepare_image({"composite": blank_canvas}) is None

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

    def test_prepare_controlnet_conditioning(self):
        """Ensure sketch is inverted and resized to 512x512 for ControlNet scribble conditioning."""
        from PIL import ImageDraw
        # White background with black center box
        img = Image.new("RGB", (100, 100), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle((40, 40, 60, 60), fill=(0, 0, 0))

        conditioning = prepare_controlnet_conditioning(img, target_size=(512, 512))
        assert conditioning.size == (512, 512)
        assert conditioning.mode == "RGB"
        # Background was white (255), after invert should be black (0, 0, 0)
        assert conditioning.getpixel((0, 0)) == (0, 0, 0)
        # Center stroke was black (0), after invert should be white (255, 255, 255)
        assert conditioning.getpixel((256, 256)) == (255, 255, 255)


class TestProcessDrawing:
    def test_empty_sketchpad(self):
        """Submitting an empty sketchpad should return None, None, 'Sketchpad is empty'."""
        img, video, status = process_drawing(None)
        assert img is None and video is None
        assert status == "Sketchpad is empty"

        img, video, status = process_drawing({})
        assert img is None and video is None
        assert status == "Sketchpad is empty"

    def test_remote_model_missing_hf_token(self, monkeypatch):
        """When HF_TOKEN and HF_KEY are missing, return 'HF_TOKEN not found'."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HF_KEY", raising=False)
        dummy_sketch = {"composite": Image.new("RGBA", (10, 10), (0, 0, 0, 255))}
        img, video, status = process_drawing(dummy_sketch, use_local_model=False)
        assert img is None and video is None
        assert status == "HF_TOKEN not found"

    def test_remote_model_success(self, monkeypatch):
        """When HF_TOKEN is set and remote API responds, return generated image and video."""
        monkeypatch.setenv("HF_TOKEN", "mock_token")
        dummy_sketch = {"composite": Image.new("RGBA", (10, 10), (0, 0, 0, 255))}
        dummy_generated_img = Image.new("RGB", (512, 512), (100, 150, 200))

        with patch("app.InferenceClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.image_to_image.return_value = dummy_generated_img
            mock_client.image_to_video.return_value = b"fake_mp4_bytes"
            mock_client_cls.return_value = mock_client

            img, video, status = process_drawing(dummy_sketch, use_local_model=False)
            assert img == dummy_generated_img
            assert video is not None
            assert "Success" in status
            mock_client.image_to_image.assert_called_once()
            mock_client.image_to_video.assert_called_once()

    def test_remote_model_failure(self, monkeypatch):
        """When InferenceClient raises an exception, return friendly error."""
        monkeypatch.setenv("HF_TOKEN", "mock_token")
        dummy_sketch = {"composite": Image.new("RGBA", (10, 10), (0, 0, 0, 255))}

        with patch("app.InferenceClient", side_effect=Exception("Connection timed out")):
            img, video, status = process_drawing(dummy_sketch, use_local_model=False)
            assert img is None and video is None
            assert "Failed to connect to inference API" in status

    def test_local_model_dispatch(self):
        """When use_local_model is True, dispatch to local_generate_pipeline."""
        dummy_sketch = {"composite": Image.new("RGBA", (10, 10), (0, 0, 0, 255))}
        dummy_img = Image.new("RGB", (512, 512), (50, 50, 50))

        with patch("app.local_generate_pipeline", return_value=(dummy_img, "path/to/video.mp4", "Success")) as mock_local:
            img, video, status = process_drawing(dummy_sketch, use_local_model=True)
            assert img == dummy_img
            assert video == "path/to/video.mp4"
            assert status == "Success"
            mock_local.assert_called_once()


class TestLocalGeneratePipeline:
    def test_local_generate_success(self):
        """Ensure local two-stage pipeline generates both image and video."""
        dummy_sketch = Image.new("RGB", (10, 10), (0, 0, 0))
        dummy_s1_img = Image.new("RGB", (512, 512), (10, 20, 30))
        dummy_frames = [Image.new("RGB", (512, 512), (10, 20, 30))]

        mock_cnet_pipe = MagicMock()
        mock_cnet_pipe.return_value = MagicMock(images=[dummy_s1_img])

        mock_svd_pipe = MagicMock()
        mock_svd_pipe.return_value = MagicMock(frames=[dummy_frames])

        with patch("app.get_controlnet_pipeline", return_value=mock_cnet_pipe), \
             patch("app.get_svd_pipeline", return_value=mock_svd_pipe), \
             patch("app.export_to_video") as mock_export:
            img, video, status = local_generate_pipeline(dummy_sketch)
            assert img == dummy_s1_img
            assert video.endswith(".mp4")
            assert "Success" in status
            mock_cnet_pipe.assert_called_once()
            mock_svd_pipe.assert_called_once()
            mock_export.assert_called_once()

    def test_local_generate_exception(self):
        """When pipeline raises an exception, return clear error message."""
        dummy_sketch = Image.new("RGB", (10, 10), (0, 0, 0))
        with patch("app.get_controlnet_pipeline", side_effect=RuntimeError("CUDA out of memory")):
            img, video, status = local_generate_pipeline(dummy_sketch)
            assert img is None and video is None
            assert "⚠️ Local Model Error: CUDA out of memory" in status


class TestGradioInterface:
    def test_interface_configuration(self):
        """Validate that the Gradio interface is properly configured without exposed text prompt."""
        assert isinstance(demo, gr.Interface)
        # Input components: Sketchpad and Checkbox (no visible prompt textbox)
        assert len(demo.input_components) == 2
        # Output components: Image, Video, Textbox
        assert len(demo.output_components) == 3

