#!/bin/bash
# Read-only service queries; writes a new evidence directory in this checkout.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

evidence_end=$(date +%s)
evidence_start=$((evidence_end - 1800))
evidence_dir="evidence/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$evidence_dir"

printf 'Collected UTC: %s\nRange: previous 30 minutes\nStep: 15 seconds\n' \
  "$(date -u +%FT%TZ)" > "$evidence_dir/context.txt"
printf '%s\n' \
  'CPU/memory from node exporter describe the shared VM, not the product container.' \
  'Feedback is user-reported. Success means a usable guess, not correct recognition.' \
  'Counters reset when the application restarts. Missing latency during no-traffic windows is expected.' \
  >> "$evidence_dir/context.txt"

docker inspect --format \
  '{{with index .NetworkSettings.Networks "group13-monitoring"}}{{.IPAddress}}{{end}}' \
  group13-drawing-game > "$evidence_dir/product-ip.txt"

curl --fail --silent --show-error --max-time 30 \
  http://127.0.0.1:13002/metrics > "$evidence_dir/application.prom"

range_query() {
  local evidence_name="$1"
  local expression="$2"
  curl --fail --silent --show-error --max-time 60 --get \
    --data-urlencode "query=$expression" \
    --data-urlencode "start=$evidence_start" \
    --data-urlencode "end=$evidence_end" \
    --data-urlencode 'step=15' \
    http://127.0.0.1:13006/api/v1/query_range \
    > "$evidence_dir/$evidence_name.json"
}

range_query scrape_health 'up{job=~"drawing_app|drawing_node"}'
range_query requests 'drawing_requests_total{job="drawing_app"}'
range_query attempts 'drawing_inference_attempts_total{job="drawing_app"}'
range_query fallbacks 'drawing_fallbacks_total{job="drawing_app"}'
range_query feedback 'drawing_feedback_total{job="drawing_app"}'
range_query in_progress 'drawing_requests_in_progress{job="drawing_app"}'
range_query model_ready 'drawing_local_model_ready{job="drawing_app"}'
range_query mean_request_seconds 'sum by (requested_mode) (rate(drawing_request_duration_seconds_sum{job="drawing_app"}[5m])) / sum by (requested_mode) (rate(drawing_request_duration_seconds_count{job="drawing_app"}[5m]))'
range_query mean_inference_seconds 'sum by (backend) (rate(drawing_inference_duration_seconds_sum{job="drawing_app"}[5m])) / sum by (backend) (rate(drawing_inference_duration_seconds_count{job="drawing_app"}[5m]))'
range_query app_memory_bytes 'process_resident_memory_bytes{job="drawing_app"}'
range_query app_cpu_cores 'rate(process_cpu_seconds_total{job="drawing_app"}[5m])'
range_query shared_vm_cpu_percent '100 * (1 - avg(rate(node_cpu_seconds_total{job="drawing_node",mode="idle"}[5m])))'
range_query shared_vm_memory_percent '100 * (1 - node_memory_MemAvailable_bytes{job="drawing_node"} / node_memory_MemTotal_bytes{job="drawing_node"})'

printf 'Evidence saved to %s\n' "$evidence_dir"
printf 'Product IP: '
cat "$evidence_dir/product-ip.txt"
