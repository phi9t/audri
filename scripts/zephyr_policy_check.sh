#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mapfile -t FILES < <(git ls-files \
  ':!:.codex-zephyr-mlsys/**' \
  ':!:.sygaldry/**')

if [[ ${#FILES[@]} -eq 0 ]]; then
  exit 0
fi

PATTERN='^\s*([>$#`-]\s*)?(uv\s+pip\s+install(\s+--system)?|pip\s+install(\s+--system)?|python3?\s+-m\s+pip\s+install)\b'

if rg -n --pcre2 "${PATTERN}" "${FILES[@]}" >/tmp/zephyr-policy-violations.txt; then
  echo "ERROR: policy violation detected. Use uv layering via launch-mlsys or scripts/zephyr_layer.sh." >&2
  cat /tmp/zephyr-policy-violations.txt >&2
  rm -f /tmp/zephyr-policy-violations.txt
  exit 1
fi

rm -f /tmp/zephyr-policy-violations.txt
echo "zephyr policy check passed"
