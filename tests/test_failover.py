from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("failure", [TimeoutError("timeout"), ConnectionError("offline"), 429, 503, "empty"])
def test_failover_and_recovery(app, monkeypatch, remote_client, drawing, failure):
    recovered = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="A Tree"))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
    )
    if isinstance(failure, int):
        failed = RuntimeError("API rejected request")
        failed.response = SimpleNamespace(status_code=failure)
        reason = f"Remote API HTTP {failure}"
    elif failure == "empty":
        failed = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" "))])
        reason = "Remote model returned an empty response"
    else:
        failed = failure
        reason = "Remote API unavailable or timed out"
    remote_client.chat.completions.create.side_effect = [failed, recovered]
    local = Mock(return_value=("A Bush", "local metrics"))
    monkeypatch.setattr(app, "local_generate", local)

    guess, metrics = app.process_drawing(drawing, incorrect_guesses=["A Flower"], return_metrics=True)
    assert guess == "A Bush"
    assert app.LOCAL_MODEL in metrics
    assert f"Automatic fallback:** {reason}" in metrics
    messages = local.call_args.args[0]
    assert messages[1]["content"][0]["image"] is drawing
    assert messages[2]["content"] == "A Flower"
    assert app.InferenceClient.call_args.kwargs["timeout"] == app.REMOTE_TIMEOUT_SECONDS

    guess, metrics = app.process_drawing(drawing, return_metrics=True)
    assert guess == "A Tree"
    assert app.REMOTE_MODEL in metrics
    assert "Automatic fallback" not in metrics
    local.assert_called_once()


def test_both_unavailable_keeps_errors_out_of_history(app, monkeypatch, drawing):
    monkeypatch.setattr(app, "InferenceClient", Mock(side_effect=ConnectionError("offline")))
    monkeypatch.setattr(app, "pipe", None)
    monkeypatch.setattr(app, "model_load_error", "Weights could not be loaded")
    result = app.make_initial_guess(drawing, False)
    assert result[0].startswith("⚠️ Both models unavailable.")
    assert "Weights could not be loaded" in result[0]
    assert "None (generation failed)" in result[1]
    assert not result[2]["visible"]
    assert result[3:] == ([], "", None, None)


def test_remote_response_without_usage_does_not_trigger_fallback(app, monkeypatch, remote_client, drawing):
    remote_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="A Tree"))], usage=None,
    )
    local = Mock()
    monkeypatch.setattr(app, "local_generate", local)
    guess, metrics = app.process_drawing(drawing, return_metrics=True)
    assert guess == "A Tree"
    assert "N/A in / N/A out" in metrics
    local.assert_not_called()
