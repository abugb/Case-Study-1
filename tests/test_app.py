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
    make_initial_guess,
    handle_correct,
    handle_incorrect,
    reset_round,
    format_history_markdown,
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

    def test_remote_model_with_incorrect_guesses(self, monkeypatch):
        """When incorrect_guesses are provided, remote messages include multi-turn feedback."""
        monkeypatch.setenv("HF_TOKEN", "mock_token")
        dummy_sketch = {"composite": Image.new("RGBA", (10, 10), (0, 0, 0, 255))}

        mock_choice = MagicMock()
        mock_choice.message.content = "A Tiger"
        mock_response = MagicMock(choices=[mock_choice])

        with patch("app.InferenceClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = process_drawing(dummy_sketch, use_local_model=False, incorrect_guesses=["A Cat"])
            assert result == "A Tiger"
            
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            assert len(messages) == 3  # Initial user message, assistant past guess, user correction
            assert messages[1]["role"] == "assistant"
            assert messages[1]["content"] == "A Cat"
            assert messages[2]["role"] == "user"
            assert "'A Cat' is incorrect" in messages[2]["content"]

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

    def test_local_model_with_incorrect_guesses(self):
        """When incorrect_guesses are provided, local messages include previous turns."""
        dummy_sketch = {"composite": Image.new("RGBA", (10, 10), (0, 0, 0, 255))}

        with patch("app.local_generate", return_value="A Fox") as mock_local:
            result = process_drawing(
                dummy_sketch,
                use_local_model=True,
                incorrect_guesses=["A Dog", "A Wolf"],
            )
            assert result == "A Fox"
            mock_local.assert_called_once()
            messages = mock_local.call_args[0][0]
            # System message + initial turn + 2 previous incorrect turns (each assistant + user) = 6
            assert len(messages) == 6
            assert messages[2]["content"] == "A Dog"
            assert messages[4]["content"] == "A Wolf"
            assert "'A Wolf' is incorrect" in messages[5]["content"]


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


class TestFeedbackLoop:
    def test_make_initial_guess_empty(self):
        """Submitting empty canvas updates guess and hides feedback buttons."""
        guess, feedback_update, history, hist_md = make_initial_guess(None, False)
        assert guess == "Sketchpad is empty"
        assert feedback_update.get("visible") is False
        assert history == []
        assert hist_md == ""

    def test_make_initial_guess_success(self):
        """Successful guess reveals feedback buttons and initializes history."""
        with patch("app.process_drawing", return_value="A Bicycle"):
            dummy_sketch = {"composite": Image.new("RGBA", (10, 10), (0, 0, 0, 255))}
            guess, feedback_update, history, hist_md = make_initial_guess(dummy_sketch, False)

            assert guess == "A Bicycle"
            assert feedback_update.get("visible") is True
            assert history == ["A Bicycle"]
            assert "A Bicycle" in hist_md

    def test_handle_correct(self):
        """Indicating correct hides feedback buttons and marks last item as correct."""
        feedback_update, history, hist_md = handle_correct(["A Cat", "A Lion"])
        assert feedback_update.get("visible") is False
        assert "(Correct!)" in hist_md

    def test_handle_incorrect(self):
        """Indicating incorrect triggers process_drawing with history and returns new guess."""
        dummy_sketch = {"composite": Image.new("RGBA", (10, 10), (0, 0, 0, 255))}
        with patch("app.process_drawing", return_value="A Leopard") as mock_proc:
            guess, feedback_update, new_history, hist_md = handle_incorrect(
                dummy_sketch, False, ["A Cat", "A Tiger"]
            )
            assert guess == "A Leopard"
            assert feedback_update.get("visible") is True
            assert new_history == ["A Cat", "A Tiger", "A Leopard"]
            assert "~~A Cat~~ (Incorrect)" in hist_md
            assert "~~A Tiger~~ (Incorrect)" in hist_md
            assert "**A Leopard** (Current Guess)" in hist_md
            mock_proc.assert_called_once_with(
                dummy_sketch, use_local_model=False, incorrect_guesses=["A Cat", "A Tiger"]
            )

    def test_reset_round(self):
        """Resetting round clears guess and hides feedback."""
        guess, feedback_update, history, hist_md = reset_round()
        assert guess == ""
        assert feedback_update.get("visible") is False
        assert history == []
        assert hist_md == ""

    def test_format_history_markdown(self):
        """Formatting markdown shows correct strike-throughs and status tags."""
        assert format_history_markdown([]) == ""
        
        md_inprogress = format_history_markdown(["Dog", "Wolf"], correct=False)
        assert "~~Dog~~ (Incorrect)" in md_inprogress
        assert "**Wolf** (Current Guess)" in md_inprogress

        md_correct = format_history_markdown(["Dog", "Wolf"], correct=True)
        assert "~~Dog~~ (Incorrect)" in md_correct
        assert "**Wolf** (Correct!)" in md_correct


class TestGradioInterface:
    def test_interface_configuration(self):
        """Validate that the Gradio interface is properly configured as gr.Blocks with input/output components."""
        assert isinstance(demo, gr.Blocks)
        assert len(demo.input_components) == 2
        assert len(demo.output_components) == 1

    def test_sketchpad_brush_configuration(self):
        """Validate that the sketchpad has color selection, color picker, and reduced thickness."""
        sketchpad = demo.input_components[0]
        assert hasattr(sketchpad, "brush")
        assert sketchpad.brush is not None
        assert sketchpad.brush.default_size <= 5
        assert len(sketchpad.brush.colors) >= 5
        assert sketchpad.brush.color_mode == "defaults"
