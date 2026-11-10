#!/bin/bash
set -eu

# Run both services as the Dockerfile's non-root app user.
prometheus-node-exporter --web.listen-address=0.0.0.0:9100 &
exporter_pid=$!

"$@" &
app_pid=$!

cleanup() {
    kill -TERM "$app_pid" "$exporter_pid" 2>/dev/null || true
    wait "$app_pid" "$exporter_pid" 2>/dev/null || true
}

trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# If either service exits, stop its companion rather than leave a partial app.
status=0
wait -n "$app_pid" "$exporter_pid" || status=$?
exit "$status"
