import base64
from io import BytesIO
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image
import pytest


def test_startup_requires_hf_token():
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    # Exercise startup in a fresh process without needing ML dependencies.
    code = """
import runpy
import sys
from unittest.mock import Mock
sys.modules['torch'] = Mock()
sys.modules['transformers'] = Mock()
runpy.run_path(sys.argv[1], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve().parents[1] / "app.py")],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "HF_TOKEN is required" in result.stderr


@pytest.mark.parametrize("token", [None, "", "   "])
@pytest.mark.parametrize("use_local", [False, True])
def test_missing_hf_token_fails_before_inference(app, monkeypatch, drawing, token, use_local):
    if token is None:
        monkeypatch.delenv("HF_TOKEN")
    else:
        monkeypatch.setenv("HF_TOKEN", token)
    local = Mock()
    monkeypatch.setattr(app, "local_generate", local)
    with pytest.raises(RuntimeError, match="HF_TOKEN is required"):
        app.process_drawing(drawing, use_local_model=use_local)
    local.assert_not_called()


def test_transparent_sketch_is_composited_onto_white(app):
    image = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
    image.putpixel((1, 0), (0, 0, 0, 255))
    prepared = app.extract_and_prepare_image({"composite": image})
    assert prepared.mode == "RGB"
    assert prepared.getpixel((0, 0)) == (255, 255, 255)
    assert prepared.getpixel((1, 0)) == (0, 0, 0)


@pytest.mark.parametrize("sketch", [None, {}, {"composite": None}])
def test_empty_sketch_does_not_run_inference(app, monkeypatch, sketch):
    local = Mock()
    monkeypatch.setattr(app, "local_generate", local)
    assert app.process_drawing(sketch) == "Sketchpad is empty"
    local.assert_not_called()


@pytest.mark.parametrize("with_metrics", [False, True])
def test_remote_request_includes_png_history_and_token_limit(app, remote_client, drawing, with_metrics):
    remote_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=" A Tiger "))],
        usage=SimpleNamespace(prompt_tokens=150, completion_tokens=5),
    )
    result = app.process_drawing(drawing, incorrect_guesses=["A Cat"], return_metrics=with_metrics)
    if with_metrics:
        guess, metrics = result
        assert app.REMOTE_MODEL in metrics
        assert "150 in / 5 out" in metrics
    else:
        guess = result
    assert guess == "A Tiger"
    request = remote_client.chat.completions.create.call_args.kwargs
    assert request["max_tokens"] == app.MAX_NEW_TOKENS == 64
    assert request["model"] == app.REMOTE_MODEL
    messages = request["messages"]
    data_url = messages[0]["content"][1]["image_url"]["url"]
    prefix, encoded = data_url.split(",", 1)
    assert prefix == "data:image/png;base64"
    with Image.open(BytesIO(base64.b64decode(encoded))) as decoded:
        assert decoded.size == drawing.size
        assert decoded.tobytes() == drawing.tobytes()
    assert messages[1] == {"role": "assistant", "content": "A Cat"}
    assert '"A Cat"' in messages[2]["content"]


@pytest.mark.parametrize("with_metrics", [False, True])
def test_manual_local_request_preserves_history(app, monkeypatch, drawing, with_metrics):
    remote = Mock()
    local = Mock(return_value=("A Fox", "local metrics"))
    monkeypatch.setattr(app, "InferenceClient", remote)
    monkeypatch.setattr(app, "local_generate", local)
    result = app.process_drawing(
        drawing, True, incorrect_guesses=["A Dog", "A Wolf"], return_metrics=with_metrics,
    )
    if with_metrics:
        guess, metrics = result
        assert app.LOCAL_MODEL in metrics
        assert "local metrics" in metrics
    else:
        guess = result
    assert guess == "A Fox"
    remote.assert_not_called()
    messages = local.call_args.args[0]
    assert messages[1]["content"][0]["image"] is drawing
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0]["image"] is drawing
    assert '"A Dog", "A Wolf"' in messages[1]["content"][1]["text"]


@pytest.mark.parametrize("output", [[], [{"generated_text": [{"content": "   "}]}]])
def test_empty_local_output_is_an_error(app, monkeypatch, output):
    monkeypatch.setattr(app, "pipe", Mock(return_value=output))
    guess, metrics = app.local_generate([])
    assert guess.startswith("⚠️ Local Model Error:")
    assert "generated messages" in guess or "blank answer" in guess
    assert metrics == ""


def test_local_inference_exception_is_reported(app, monkeypatch):
    monkeypatch.setattr(app, "pipe", Mock(side_effect=RuntimeError("Inference failed")))
    guess, metrics = app.local_generate([])
    assert guess == "⚠️ Local Model Error: Inference failed"
    assert metrics == ""


def test_local_generation_reports_output_tokens(app, monkeypatch):
    pipe = Mock(return_value=[{"generated_text": [{"content": " A Cat "}]}])
    pipe.tokenizer.encode.return_value = [1, 2, 3]
    monkeypatch.setattr(app, "pipe", pipe)
    guess, metrics = app.local_generate([])
    assert guess == "A Cat"
    assert "N/A in / 3 out" in metrics
    assert pipe.call_args.kwargs["generate_kwargs"] == {"max_new_tokens": 64, "do_sample": False}


@pytest.mark.parametrize("handler", ["make_initial_guess", "handle_incorrect"])
@pytest.mark.parametrize("changed", [False, True])
def test_history_tracks_the_submitted_drawing(app, monkeypatch, drawing, handler, changed):
    previous = ["A Cat", "A Dog"]
    previous_hash = app.get_drawing_hash(Image.new("RGB", drawing.size, "red") if changed else drawing)
    process = Mock(return_value=("A Tree", "metrics"))
    monkeypatch.setattr(app, "process_drawing", process)
    result = getattr(app, handler)(drawing, False, previous, previous_hash)
    guess, metrics, feedback, history, markdown, current_hash, cached = result
    expected_previous = [] if changed else previous
    assert (guess, metrics) == ("A Tree", "metrics")
    assert feedback["visible"]
    assert history == expected_previous + [guess]
    assert "**A Tree** (Current Guess)" in markdown
    if not changed:
        assert "~~A Cat~~ (Incorrect)" in markdown
    assert current_hash == app.get_drawing_hash(drawing)
    assert cached is drawing
    assert process.call_args.kwargs["incorrect_guesses"] == expected_previous


def test_guess_again_uses_cached_drawing(app, monkeypatch, drawing):
    process = Mock(return_value=("A Tree", ""))
    monkeypatch.setattr(app, "process_drawing", process)
    result = app.handle_incorrect(None, True, ["A Cat"], "drawing-hash", drawing)
    assert process.call_args.args[0] is drawing
    assert result[3] == ["A Cat", "A Tree"]


def test_failed_retry_preserves_history(app, monkeypatch, drawing):
    monkeypatch.setattr(app, "process_drawing", Mock(return_value=("⚠️ Local Model Error: offline", "")))
    result = app.handle_incorrect(drawing, True, ["A Cat"], app.get_drawing_hash(drawing), drawing)
    assert result[2]["visible"]
    assert result[3] == ["A Cat"]


def test_correct_feedback_marks_last_guess(app):
    feedback, history, markdown = app.handle_correct(["A Cat", "A Tree"])
    assert not feedback["visible"]
    assert history == ["A Cat", "A Tree"]
    assert "~~A Cat~~ (Incorrect)" in markdown
    assert "**A Tree** (Correct!)" in markdown


def test_empty_submission_and_reset_clear_round(app):
    for result in (app.make_initial_guess(None, False), app.handle_incorrect(None, False), app.reset_round()):
        assert not result[2]["visible"]
        assert result[1] == ""
        assert result[3:] == ([], "", None, None)


def test_gradio_endpoint_and_reset_events_are_wired(app):
    endpoint = next(fn for fn in app.demo.fns.values() if fn.api_name == "process_drawing")
    assert endpoint.inputs == [app.sketchpad, app.use_local_model]
    assert endpoint.outputs == [app.guess_output]
    events = {(component, event): fn for fn in app.demo.fns.values() for component, event in fn.targets}
    assert (app.sketchpad._id, "change") not in events
    for target in ((app.sketchpad._id, "clear"), (app.use_local_model._id, "change")):
        assert events[target].fn is app.reset_round
        assert events[target].outputs == app.feedback_outputs
