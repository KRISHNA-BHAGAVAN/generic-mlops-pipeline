#!/bin/sh
# ─────────────────────────────────────────────────────────────────
# Prometheus entrypoint — handles env var substitution in configs.
#
# Prometheus only natively expands env vars in external_labels.
# For remote_write URLs, auth, etc. we need to pre-process the
# config file before handing it to Prometheus.
#
# If GRAFANA_CLOUD_PROM_URL is set → uses prometheus.cloud.yml
# Otherwise → uses prometheus.yml (local-only mode)
# ─────────────────────────────────────────────────────────────────

set -e

TEMPLATE_DIR="/etc/prometheus"
OUTPUT="/tmp/prometheus.yml"

# Determine which config template to use
if [ -n "${GRAFANA_CLOUD_PROM_URL}" ]; then
    TEMPLATE="${TEMPLATE_DIR}/prometheus.cloud.yml"
    echo "[entrypoint] Cloud mode — remote_write enabled"
else
    TEMPLATE="${TEMPLATE_DIR}/prometheus.yml"
    echo "[entrypoint] Local mode — no remote_write"
fi

# Substitute environment variables using sed
# (prom/prometheus is busybox-based, no envsubst available)
cp "${TEMPLATE}" "${OUTPUT}"
sed -i \
    -e "s|\${GRAFANA_CLOUD_PROM_URL}|${GRAFANA_CLOUD_PROM_URL}|g" \
    -e "s|\${GRAFANA_CLOUD_PROM_USERNAME}|${GRAFANA_CLOUD_PROM_USERNAME}|g" \
    -e "s|\${PROMETHEUS_DEVELOPER}|${PROMETHEUS_DEVELOPER:-anonymous}|g" \
    "${OUTPUT}"

echo "[entrypoint] Config written to ${OUTPUT}"

# Start Prometheus with the processed config
exec /bin/prometheus \
    --config.file="${OUTPUT}" \
    "$@"
