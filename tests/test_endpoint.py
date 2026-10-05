"""Live checks: GRADIO_TEST_URL=http://127.0.0.1:8013 python -m pytest -m e2e."""
import os

from gradio_client import Client, handle_file
from PIL import Image, ImageDraw
import pytest

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def client():
    url = os.environ.get("GRADIO_TEST_URL")
    if not url:
        pytest.skip("Set GRADIO_TEST_URL to enable live endpoint tests")
    return Client(url, token=os.environ.get("HF_TOKEN"), verbose=False)


def test_endpoint_empty_sketch(client):
    payload = {"background": None, "layers": [], "composite": None}
    assert client.predict(sketch=payload, use_local_model=True, api_name="/process_drawing") == "Sketchpad is empty"


def test_endpoint_local_drawing(client, tmp_path):
    image = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(image).ellipse((40, 40, 160, 160), fill="black")
    path = tmp_path / "drawing.png"
    image.save(path)
    payload = {"background": None, "layers": [], "composite": handle_file(str(path))}
    response = client.predict(sketch=payload, use_local_model=True, api_name="/process_drawing")
    assert response.strip()
    assert response != "Sketchpad is empty"
    assert not response.startswith("⚠️"), response
