#!/usr/bin/env bash
# Capture the entire job, including interpreter/preflight failures, without Git.
# Usage: bash scripts/run_logged.sh COMMAND [ARG ...]
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_root="${ACP_LOG_DIR:-$repo_root/tmp_files/acp_logs}"
mkdir -p "$log_root"
log_root="$(cd "$log_root" && pwd)"
log_file="$(mktemp "$log_root/acp-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.log")"
export MERL_CONSOLE_LOG="$log_file"
started=$SECONDS
printf '[acp] console log: %s\n' "$log_file"

# Capture both statuses immediately: tee must not hide a failed training command.
set +e
(
  set -e
  trap 'result=$?; printf "[acp] finished_utc=%s command_exit_code=%s elapsed_seconds=%s\n" "$(date -u +%FT%TZ)" "$result" "$((SECONDS - started))"' EXIT
  printf '[acp] started_utc=%s log_file=%s\n' "$(date -u +%FT%TZ)" "$log_file"
  printf '[acp] command:'
  printf ' %q' "$@"
  printf '\n'
  if [[ $# -eq 0 ]]; then
    printf '[acp] error: a command is required\n' >&2
    exit 2
  fi
  "$@"
) 2>&1 | tee "$log_file"
statuses=("${PIPESTATUS[@]}")
set -e
result="${statuses[0]}"
[[ "$result" -ne 0 ]] || result="${statuses[1]}"
printf '{"exit_code":%s,"command_exit_code":%s,"logging_exit_code":%s,"elapsed_seconds":%s}\n' \
  "$result" "${statuses[0]}" "${statuses[1]}" "$((SECONDS - started))" > "${log_file%.log}.status.json"
exit "$result"
