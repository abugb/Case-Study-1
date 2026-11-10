"""Offline checks of the app's actual handlers; no model, token, or API calls.

Load function definitions and metric declarations from app.py without running
model initialization or building the Gradio UI. Remote UI verification is still
required after deployment.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from urllib.request import urlopen


def load_handlers():
    tree = ast.parse(Path(__file__).with_name("app.py").read_text(encoding="utf-8"))
    nodes = []
    excluded = {"gradio", "torch", "transformers", "huggingface_hub", "dotenv"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            if all(alias.name not in excluded for alias in node.names):
                nodes.append(node)
        elif isinstance(node, ast.ImportFrom):
            if node.module not in excluded:
                nodes.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.For)):
            nodes.append(node)
        elif isinstance(node, ast.Assign):
            if not any(isinstance(target, ast.Name) and target.id == "brush" for target in node.targets):
                nodes.append(node)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), namespace)
    namespace["gr"] = SimpleNamespace(update=lambda **kwargs: kwargs)
    namespace["require_hf_token"] = lambda: "offline-test-placeholder"
    return namespace


APP = load_handlers()


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.image = APP["Image"].new("RGB", (8, 8), "white")
        self.pipe = Mock(return_value=[{"generated_text": [{"content": "cat"}]}])
        self.pipe.tokenizer.encode.return_value = [1]
        APP["pipe"] = self.pipe
        self.remote = Mock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="dog"))],
            usage=None,
        ))
        APP["InferenceClient"] = lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=self.remote)))

    def value(self, metric, *labels):
        collector = APP[metric]
        if labels:
            collector = collector.labels(*labels)
        return collector._value.get()

    def assert_increment(self, metric, labels, action, increment=1):
        before = self.value(metric, *labels)
        result = action()
        self.assertEqual(self.value(metric, *labels) - before, increment)
        return result

    def test_remote_success_does_not_invoke_local(self):
        result = self.assert_increment("ATTEMPTS", ("api", "success"),
            lambda: APP["process_drawing"](self.image))
        self.assertEqual(result, "dog")
        self.pipe.assert_not_called()
        self.assertEqual(self.value("IN_PROGRESS", "api"), 0)

    def test_local_success(self):
        result = self.assert_increment("REQUESTS", ("local", "success"),
            lambda: APP["process_drawing"](self.image, True))
        self.assertEqual(result, "cat")
        self.remote.assert_not_called()

    def test_timeout_fallback_counts_one_request_two_attempts(self):
        self.remote.side_effect = TimeoutError("simulated")
        targets = [("REQUESTS", ("api", "success")),
                   ("ATTEMPTS", ("api", "error")),
                   ("ATTEMPTS", ("local", "success")),
                   ("FALLBACKS", ("timeout",))]
        before = [self.value(metric, *labels) for metric, labels in targets]
        result = APP["process_drawing"](self.image, return_metrics=True)
        self.assertEqual(result[0], "cat")
        self.assertIn("Automatic fallback", result[1])
        for (metric, labels), old in zip(targets, before):
            self.assertEqual(self.value(metric, *labels) - old, 1)

    def test_empty_remote_output_falls_back(self):
        self.remote.return_value.choices[0].message.content = "  "
        self.assert_increment("FALLBACKS", ("empty_response",),
            lambda: APP["process_drawing"](self.image))

    def test_both_backends_fail_and_gauge_recovers(self):
        self.remote.side_effect = RuntimeError("simulated")
        self.pipe.side_effect = RuntimeError("simulated")
        result = self.assert_increment("REQUESTS", ("api", "error"),
            lambda: APP["process_drawing"](self.image))
        self.assertTrue(APP["is_error_response"](result))
        self.assertEqual(self.value("IN_PROGRESS", "api"), 0)

    def test_empty_drawing_from_ui_is_counted_without_inference(self):
        self.assert_increment("REQUESTS", ("local", "invalid_input"),
            lambda: APP["make_initial_guess"](None, True))
        self.pipe.assert_not_called()
        self.remote.assert_not_called()

    def test_feedback_tracks_fallback_backend_and_consumes_state(self):
        self.remote.side_effect = TimeoutError("simulated")
        result = APP["make_initial_guess"](self.image, False)
        backend = result[-1]
        self.assertEqual(backend, "local")
        feedback = self.assert_increment("FEEDBACK", ("local", "correct"),
            lambda: APP["handle_correct"](result[3], backend))
        self.assertIsNone(feedback[-1])
        self.assert_increment("FEEDBACK", ("local", "correct"),
            lambda: APP["handle_correct"](result[3], feedback[-1]), increment=0)

    def test_incorrect_feedback_then_next_guess(self):
        result = APP["make_initial_guess"](self.image, True)
        next_result = self.assert_increment("FEEDBACK", ("local", "incorrect"),
            lambda: APP["handle_incorrect"](
                self.image, True, result[3], result[5], result[6], result[7]))
        self.assertEqual(len(next_result[3]), 2)
        self.assertEqual(next_result[-1], "local")

    def test_quota_and_rate_limit_are_distinguished(self):
        self.assertEqual(APP["fallback_category"](
            SimpleNamespace(response=SimpleNamespace(status_code=402))), "quota")
        self.assertEqual(APP["fallback_category"](
            SimpleNamespace(response=SimpleNamespace(status_code=429))), "other")

    def test_request_timer_includes_lock_wait_but_inference_timer_does_not(self):
        from unittest.mock import patch
        clock = [0.0]

        class WaitingLock:
            def __enter__(self):
                clock[0] += 10

            def __exit__(self, *args):
                return False

        def generate(**kwargs):
            clock[0] += 2
            return [{"generated_text": [{"content": "cat"}]}]

        self.pipe.side_effect = generate
        request_before = APP["REQUEST_DURATION"].labels("local")._sum.get()
        inference_before = APP["INFERENCE_DURATION"].labels("local")._sum.get()
        with patch.object(APP["time"], "perf_counter", side_effect=lambda: clock[0]), \
                patch.dict(APP, {"local_model_lock": WaitingLock()}):
            APP["process_drawing"](self.image, True)
        self.assertEqual(APP["REQUEST_DURATION"].labels("local")._sum.get() - request_before, 12)
        self.assertEqual(APP["INFERENCE_DURATION"].labels("local")._sum.get() - inference_before, 2)

    def test_http_metrics_endpoint_exposes_all_eight_families(self):
        server, thread = APP["start_http_server"](0, addr="127.0.0.1")
        try:
            with urlopen(f"http://127.0.0.1:{server.server_port}/metrics", timeout=5) as response:
                body = response.read().decode()
            # The Python client also emits companion *_created timestamp gauges.
            families = [line for line in body.splitlines()
                        if line.startswith("# TYPE drawing_") and not line.split()[2].endswith("_created")]
            self.assertEqual(len(families), 8)
            self.assertIn("drawing_request_duration_seconds_bucket", body)
            self.assertIn("drawing_local_model_ready", body)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
