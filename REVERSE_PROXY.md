# Public access through one ngrok hostname

The intended route is ngrok -> VM loopback port 13001 -> group13-proxy
(Nginx port 80). The proxy and both applications share group13-monitoring.

- `/` and other product paths go to group13-drawing-game:7860.
- `/grafana` redirects to `/grafana/`.
- `/grafana/` and its descendants go to group13-grafana:3000 with the path preserved.
- Prometheus continues scraping the product directly; it is not publicly proxied.

Grafana must use these settings alongside its existing volume, data-source
provisioning, WAL setting, and resource limits:

```text
GF_SERVER_ROOT_URL=https://haggler-eatery-autopilot.ngrok-free.dev/grafana/
GF_SERVER_SERVE_FROM_SUB_PATH=true
GF_DATABASE_WAL=true
```

The existing admin account and dashboard remain in group13-grafana-data. Keep
prior Grafana containers stopped because they reference the same database.

nginx.conf uses Docker DNS with short caching so that replacing an application
container does not permanently leave the proxy targeting its old IP address.
It forwards the external scheme, host, and WebSocket upgrade headers, disables
response buffering for Gradio streaming, and allows 20 MiB request bodies.

Validate nginx.conf using `nginx -t` in the official Nginx image before deployment.
After deploying, verify the product at VM port 13001 and Grafana health at
`http://127.0.0.1:13001/grafana/api/health`. Then start only the named `product`
tunnel from ngrok-tunnels.yml; its upstream is now port 13001.

Public verification must include a drawing submission, Grafana login, dashboard
loading and refresh, and an Explore query. A syntax check alone does not verify
these behaviors. After the subpath change, access Grafana through the public
`/grafana/` URL; its old standalone root URL may redirect to the public subpath.

The product link belongs in the class sheet. Include the Grafana dashboard link
and screenshots in the report, rather than labeling Grafana as a second product.
