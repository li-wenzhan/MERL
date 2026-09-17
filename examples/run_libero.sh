#!/usr/bin/env bash
# All online modes/jobs share the same validated Python entrypoint.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python -m merl.launch "$@"
