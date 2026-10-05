"""Manual CPU integration test: python tests/smoke_local_model.py.

Loads real weights once; checks drawing recognition, API-failure fallback,
and incorrect-guess feedback without calling a paid inference API.
"""
import json
import os
from pathlib import Path
import resource
import sys
import time
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Test-only credential: every remote request below is mocked; none is sent.
os.environ["HF_TOKEN"] = "smoke-test-token"
import app


def drawing(subject):
    image = Image.new("RGB", (384, 384), "white")
    pen = ImageDraw.Draw(image)
    if subject == "house":
        pen.rectangle((80, 165, 300, 330), fill="lightyellow", outline="black", width=5)
        pen.polygon([(55, 165), (190, 50), (325, 165)], fill="red", outline="black", width=5)
        pen.rectangle((165, 240, 215, 330), fill="brown", outline="black", width=4)
        for x in (110, 240):
            pen.rectangle((x, 200, x + 35, 235), fill="lightblue", outline="black", width=4)
    else:
        pen.rectangle((168, 190, 215, 340), fill="brown", outline="black", width=4)
        for box in ((90, 95, 220, 235), (170, 95, 300, 235), (125, 45, 260, 200)):
            pen.ellipse(box, fill="green", outline="black", width=3)
    return image


if __name__ == "__main__":
    assert app.pipe is not None, app.model_load_error
    assert app.pipe.model.device.type == "cpu"
    for subject, use_local, size in (("house", True, 1024), ("tree", False, 384)):
        start = time.perf_counter()
        with patch("app.InferenceClient", Mock(side_effect=ConnectionError("test outage"))):
            result = app.make_initial_guess(drawing(subject).resize((size, size)), use_local)
        guess, metrics = result[:2]
        assert not app.is_error_response(guess), guess
        assert subject in guess.lower(), (subject, guess)
        assert result[2]["visible"] and result[3] == [guess]
        assert app.LOCAL_MODEL in metrics
        if not use_local:
            assert "Automatic fallback" in metrics
        print(json.dumps({"subject": subject, "size": size, "guess": guess,
                          "seconds": round(time.perf_counter() - start, 2),
                          "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}), flush=True)

    # A deliberately wrong previous answer checks multimodal chat history.
    image = drawing("house")
    result = app.handle_incorrect(image, True, ["A banana"], app.get_drawing_hash(image), image)
    assert not app.is_error_response(result[0]), result[0]
    assert "house" in result[0].lower(), result[0]
    assert result[3] == ["A banana", result[0]]
    print("Incorrect-guess feedback passed:", result[0], flush=True)
