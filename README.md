# Case Study 3: Drawing Game Deployment and Monitoring

This branch extends `Case-Study-2` with Docker deployment, application metrics,
node exporter, Prometheus, Grafana, and one public ngrok entry point.
The Gradio drawing game retains both remote API and local CPU inference,
including automatic local fallback when remote inference fails.

## Deployment layout

All four containers join the `group13-monitoring` Docker network:

| Container | Role | VM loopback mapping |
|---|---|---|
| group13-drawing-game | Gradio, Python metrics, node exporter | 13000:7860, 13002:8000, 13004:9100 |
| group13-prometheus | Metrics collection and storage | 13006:9090 |
| group13-grafana | Dashboard | 13007:3000 |
| group13-proxy | Nginx public path routing | 13001:80 |

The public product is at
https://haggler-eatery-autopilot.ngrok-free.dev/ and Grafana is at
https://haggler-eatery-autopilot.ngrok-free.dev/grafana/ while the tunnel is online.
Grafana requires its configured login. See [REVERSE_PROXY.md](REVERSE_PROXY.md)
for routing details and required Grafana subpath settings.

## Image and runtime

Build with `docker build -t group13/drawing-game:cs3-node .`.
The Dockerfile installs CPU inference dependencies and the
`prometheus-node-exporter` apt package. `start.sh` runs the app and exporter
as the non-root app user and shuts down both when either exits.

Supply `HF_TOKEN` at runtime, not during image building. The existing app
requires a nonempty token even in local mode. The local model is
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`; remote requests use
`Qwen/Qwen3-VL-235B-A22B-Instruct`. Initial startup downloads local model
weights; persist them using `group13-hf-cache` mounted at
`/home/app/.cache/huggingface`. The deployed product uses two CPUs and 4 GiB RAM.

## Monitoring configuration

- `prometheus.yml`: application and node scrape jobs every 15 seconds.
- `grafana-datasource.yml`: Prometheus data source provisioning.
- `group13-dashboard.json`: importable dashboard for the provisioned data source.
- `nginx.conf`: product at `/`, Grafana at `/grafana/`, streaming and WebSockets.
- `ngrok-tunnels.yml`: one HTTPS tunnel to proxy port 13001; credentials live
  separately in the account's ngrok configuration and must not be committed.
- `collect-evidence.sh`: exports a metrics snapshot and 30 minutes of series.

Eight custom application metric families cover completed requests, request
latency, backend attempts, inference latency, fallback reasons, active requests,
local-model readiness, and user feedback. Successful API-selected requests can
be served by local fallback; inspect backend attempt and fallback metrics before
claiming remote inference success. Feedback is user-reported, not benchmark accuracy.
Node CPU/memory measurements describe the shared VM, not container limits.
Process metrics describe the Python application process.

## Offline verification

Install the CI test dependencies (ML weights are not needed):

```bash
python -m pip install pytest 'gradio>=6.26,<7' pillow huggingface_hub python-dotenv 'prometheus-client>=0.21,<1'
python -m pytest -m "not e2e" -q
python -m unittest discover -s . -p test_metrics.py -v
```

The metric tests simulate inference and check request/fallback accounting,
feedback attribution, timing boundaries, and HTTP exposition. Existing unit
tests also mock ML runtimes. Run these suites in separate processes to keep
metric registries isolated. Live Gradio checks remain opt-in through
`GRADIO_TEST_URL` and `pytest -m e2e`; they are not run by the offline commands.

Do not commit `.env`, account credentials, raw evidence, runtime databases,
model caches, or downloaded binaries. Persistent volumes hold model downloads,
Prometheus history, and Grafana accounts/dashboards. Container restart policies
are configured on the deployed VM; the background ngrok process survives SSH
logout but is not configured to restart automatically after a VM reboot.
