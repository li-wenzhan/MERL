#!/usr/bin/env bash
# One ACP allocation, sequential short training and full real-environment evaluation.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
exec bash scripts/run_logged.sh python -u -m merl.presentation_run "$@"
