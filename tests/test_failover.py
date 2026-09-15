from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

import app


@pytest.mark.parametrize("failure", [TimeoutError("timeout"), ConnectionError("offline"), 429, 503, "empty"])
def test_failover_and_recovery(monkeypatch, failure):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    recovered = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="A Tree"))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
    )
    if isinstance(failure, int):
        error = RuntimeError("API rejected request")
        error.response = SimpleNamespace(status_code=failure)
    elif failure == "empty":
        error = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" "))])
    else:
        error = failure
    client = Mock()
    client.chat.completions.create.side_effect = [error, recovered]
    factory = Mock(return_value=client)
    monkeypatch.setattr(app, "InferenceClient", factory)
    local = Mock(return_value=("A Bush", "local metrics"))
    monkeypatch.setattr(app, "local_generate", local)
    drawing = Image.new("RGB", (8, 8), "white")

    guess, metrics = app.process_drawing(drawing, incorrect_guesses=["A Flower"], return_metrics=True)
    assert guess == "A Bush"
    assert app.LOCAL_MODEL in metrics
    assert "Automatic fallback" in metrics
    messages = local.call_args.args[0]
    assert messages[1]["content"][0]["image"] is drawing
    assert messages[2]["content"] == "A Flower"
    assert factory.call_args.kwargs["timeout"] == app.REMOTE_TIMEOUT_SECONDS
    print(f"{failure}: Local -> {guess}")

    guess, metrics = app.process_drawing(drawing, return_metrics=True)
    assert guess == "A Tree"
    assert app.REMOTE_MODEL in metrics
    assert "Automatic fallback" not in metrics
    local.assert_called_once()
    print(f"Recovered: Remote -> {guess}")


def test_both_unavailable_keeps_errors_out_of_history(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(app, "pipe", None)
    result = app.make_initial_guess(Image.new("RGB", (8, 8)), False)
    assert result[0].startswith("⚠️ Both models unavailable.")
    assert "None (generation failed)" in result[1]
    assert result[2]["visible"] is False
    assert result[3] == []


def test_manual_local_does_not_call_remote(monkeypatch):
    remote = Mock()
    monkeypatch.setattr(app, "InferenceClient", remote)
    monkeypatch.setattr(app, "local_generate", Mock(return_value=("A Tree", "")))
    _, metrics = app.process_drawing(Image.new("RGB", (8, 8)), True, return_metrics=True)
    assert app.LOCAL_MODEL in metrics
    remote.assert_not_called()
