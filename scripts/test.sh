#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
bend_bin=${BEND:-bend}
if ! command -v "$bend_bin" >/dev/null 2>&1; then
  bend_bin="$HOME/.bend/bin/bend"
fi
case "${1:-}" in
  ''|--native) ;;
  *) echo 'Usage: ./scripts/test.sh [--native]' >&2; exit 2 ;;
esac
"$bend_bin" PROOF.bend
"$bend_bin" tests/test.bend
"$bend_bin" examples/linear_regression.bend
if [ "${1:-}" = --native ]; then
  mkdir -p build
  "$bend_bin" tests/test.bend -o build/tests
  ./build/tests --threads 4
fi
