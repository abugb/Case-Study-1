# tests written by AI
import os
import time
import tempfile
import pytest
from PIL import Image, ImageDraw
from huggingface_hub import HfApi, get_token
from gradio_client import Client, handle_file

SPACE_REPO_ID = os.environ.get("HF_SPACE_ID", "abugb/Case-Study-1")
TIMEOUT_SECONDS = int(os.environ.get("SPACE_TIMEOUT", "600"))
POLL_INTERVAL_SECONDS = 10

pytestmark = pytest.mark.e2e


def get_auth_token():
    """Retrieve token from environment or local cache."""
    return os.environ.get("HF_TOKEN") or get_token()


def wait_for_space_ready(repo_id: str, token: str | None, timeout: int = TIMEOUT_SECONDS) -> None:
    """Poll Space runtime stage until it is RUNNING or enters an error state."""
    api = HfApi(token=token)
    start_time = time.time()
    last_stage = None

    print(f"\nWaiting for Hugging Face Space '{repo_id}' to become RUNNING...")
    while time.time() - start_time < timeout:
        try:
            runtime = api.get_space_runtime(repo_id)
            stage = runtime.stage
        except Exception as e:
            print(f"Warning fetching space runtime: {e}")
            stage = "FETCH_FAILED"

        if stage != last_stage:
            print(f"Space stage changed: {stage}")
            last_stage = stage

        if stage == "RUNNING":
            raw = getattr(runtime, "raw", {})
            domains = raw.get("domains", []) if isinstance(raw, dict) else []
            if domains and domains[0].get("stage") != "READY":
                print(f"Space domain is not READY yet: {domains[0].get('stage')}")
            else:
                print(f"Space '{repo_id}' is RUNNING and READY.")
                return

        if stage in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "PAUSED", "STOPPED"):
            raise RuntimeError(f"Hugging Face Space '{repo_id}' entered terminal state: {stage}")

        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"Hugging Face Space '{repo_id}' did not reach RUNNING within {timeout}s (last stage: {last_stage})"
    )


@pytest.fixture(scope="module")
def hf_client():
    token = get_auth_token()
    if not token:
        pytest.skip("HF_TOKEN not found in environment or local cache; skipping E2E tests.")
    wait_for_space_ready(SPACE_REPO_ID, token)
    client = Client(SPACE_REPO_ID, token=token)
    return client


@pytest.fixture(scope="module")
def sample_sketch():
    """Creates a temporary image with a drawn circle."""
    img = Image.new("RGBA", (200, 200), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((40, 40, 160, 160), fill=(0, 0, 0, 255))

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        img.save(tmp_file.name)
        tmp_path = tmp_file.name

    payload = {
        "background": None,
        "layers": [],
        "composite": handle_file(tmp_path),
    }

    yield payload

    if os.path.exists(tmp_path):
        os.remove(tmp_path)


class TestSpaceE2E:
    def test_space_empty_sketchpad(self, hf_client):
        """Empty sketchpad payload should return None, None, 'Sketchpad is empty'."""
        empty_payload = {"background": None, "layers": [], "composite": None}
        response = hf_client.predict(
            sketch=empty_payload,
            use_local_model=False,
            api_name="/process_drawing",
        )
        img, video, status = response
        assert img is None
        assert video is None
        assert status == "Sketchpad is empty"

    def test_space_remote_model(self, hf_client, sample_sketch):
        """Test remote inference dispatch and provider response handling."""
        response = hf_client.predict(
            sketch=sample_sketch,
            use_local_model=False,
            api_name="/process_drawing",
        )
        print(f"\nRemote Model Response: {response}")
        img, video, status = response
        assert "HF_TOKEN not found" not in status
        assert "Sketchpad is empty" not in status
        assert isinstance(status, str)
        # On Hugging Face Serverless API, multi-modal I2V diffusion tasks require
        # dedicated endpoints; verify the remote handler produces outputs or
        # captures the provider endpoint response gracefully without unhandled crashes.
        if img is None:
            assert "Failed to connect to inference API" in status
        else:
            assert video is not None


    def test_space_local_model_zerogpu(self, hf_client, sample_sketch):
        """Test local ZeroGPU two-stage diffusion pipeline."""
        response = hf_client.predict(
            sketch=sample_sketch,
            use_local_model=True,
            api_name="/process_drawing",
        )
        print(f"\nZeroGPU Local Model Response: {response}")
        img, video, status = response
        assert not status.startswith("⚠️ Local Model Error"), f"ZeroGPU model error: {status}"
        assert "Sketchpad is empty" not in status
        assert img is not None
        assert video is not None


