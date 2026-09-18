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
"$bend_bin" bend_ml.bend
"$bend_bin" tests/test.bend
"$bend_bin" examples/linear_regression.bend
# Exercise the README verbatim as an external consumer, including import paths.
mkdir -p build/readme-example/vendor/bend_ml
cp -R src build/readme-example/vendor/bend_ml/
awk '/^```python$/ {printing=1; next} /^```$/ && printing {exit} printing {print}' README.md > build/readme-example/main.bend
readme_output=$("$bend_bin" build/readme-example/main.bend)
if [ "$readme_output" != 11 ]; then
  echo "README example: expected 11, got $readme_output" >&2
  exit 1
fi
echo 'PASS: vendored README example'
hub_output=$("$bend_bin" examples/from_hub.bend)
if [ "$hub_output" != 11 ]; then
  echo "BendHub example: expected 11, got $hub_output" >&2
  exit 1
fi
echo 'PASS: published BendHub example'
if [ "${1:-}" = --native ]; then
  mkdir -p build
  "$bend_bin" tests/test.bend -o build/tests
  ./build/tests --threads 4
fi
