---
title: Case Study 1
emoji: 🐠
colorFrom: pink
colorTo: pink
sdk: gradio
sdk_version: 6.26.0
python_version: '3.12'
app_file: app.py
pinned: false
short_description: Interface where an LLM guesses your drawing
---

## Running in the CS2 remote container

From the local checkout, run `bash CS2/connect.sh` and keep the SSH session
open. The script forwards local port 8013 to port 8013 in the container. In that session:

```bash
cd ~/cs553-product
source .venv/bin/activate
python -m pip install -r requirements.txt
# Required: place HF_TOKEN in .env beside app.py (or export it in this shell).
# python-dotenv loads .env when the app starts.
python app.py
```

Open http://localhost:8013 on your own computer. Gradio defaults to port 8013
(overridable with `GRADIO_SERVER_PORT`). Hosted inference requires an HF_TOKEN
with provider access and sufficient quota. Missing or blank HF_TOKEN stops
startup with an error, including when local inference is desired. Local fallback
is used only for API failures or empty responses when a token is configured.

### Local CPU model

The local model is `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. It runs on
CPU in float32 with SDPA attention; CUDA, FlashAttention, and Hugging Face
Spaces are not required. `num2words` is required by its processor. The first
startup downloads the model to the Hugging Face cache.

For the two-core, 4 GiB CS2 container, the app limits PyTorch to two threads
and serializes local inference. Both local and remote generation use
`MAX_NEW_TOKENS = 64` (a maximum, not a required response length).

Drawings are passed as PIL images without lossy encoding. Local image
splitting is explicitly disabled: the processor resizes the complete drawing
to one model-sized image. The remote API receives one complete PNG; its
internal preprocessing is managed by the inference provider.

Remote token counts come from API usage. Local output tokens are counted
from the returned text; local input tokens show `N/A` because the pipeline
does not report the combined image and prompt count.

A CPU stress test forcing all 64 output tokens with splitting disabled used
about 2.57 GiB peak process RSS and took 7.1 seconds. Ordinary short object-name
responses can finish sooner. These synthetic-drawing tests are not a general
accuracy benchmark or a guarantee for arbitrarily long conversation histories.

The sketchpad, incorrect-guess history, remote model, and automatic fallback
are retained. Startup model-loading errors are logged and shown when local
inference is requested, so remote inference can still be used.

Run the real-model application smoke test (downloads weights if uncached):

```bash
python tests/smoke_local_model.py
```

Run unit tests without downloading a model:

```bash
python -m pytest -m "not e2e" -q
```

Unit tests mock the ML runtimes and inference API. Their test credential is
scoped to the unit-test fixtures and is never used by live endpoint tests.
The CI job installs only the dependencies needed for these offline tests.

Test the running container's Gradio endpoint from inside the container:

```bash
GRADIO_TEST_URL=http://127.0.0.1:8013 python -m pytest -m e2e -q
```

`GRADIO_TEST_URL` also accepts a Hugging Face Space ID for hosted checks.
These live tests exercise empty sketch submission and local CPU inference
through the API. Without `GRADIO_TEST_URL` they skip; a configured endpoint
that is unavailable fails the tests. The real-model smoke test above also
checks recognition, fallback during a simulated API outage, and guess-again
history without making paid inference requests.
