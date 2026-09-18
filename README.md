# bend-ml

A small machine-learning experiment in [Bend 2](https://bend-lang.com/).

One model, a few shared building blocks, and examples you can change:

- **Linear regression:** ordinary least squares for one feature, with a fitted intercept.
- **Batches:** reusable trees, list conversion, mapping, and parallel reductions.
- **Statistics:** mergeable centered moments for training.
- **Metrics:** mean squared error and floating-point comparison helpers.

Tested with **Bend 2.0.5**. This is an early experiment, with no performance claims or external library dependencies beyond Bend's `Base`. It does not use legacy Bend 1 / HVM.

## Install and try it

Install Bend on macOS or Linux using its official installer:

```sh
curl -fsSL https://bend-lang.com/install.sh | sh
export PATH="$HOME/.bend/bin:$PATH"
bend --version
```

Clone this library and run the small example:

```sh
git clone https://github.com/lvtd-llc/bend-ml.git
cd bend-ml
bend examples/linear_regression.bend
```

Expected output:

```text
Fitted y = 2x + 1
Prediction at x=5 (expected 11): 11
Prediction at x=10 (expected 21): 21
```

`bend file.bend` uses the sequential JavaScript backend. No GPU or C compiler is needed for this first example; the Bend installer also installs Bun if needed.

## Use it in your own Bend project

From your project's root, place the library in `vendor/`:

```sh
mkdir -p vendor
git clone https://github.com/lvtd-llc/bend-ml.git vendor/bend-ml
```

Save this as `main.bend` in your project root:

```python
import Base
import ./vendor/bend-ml/src/batch.bend as Batch
import ./vendor/bend-ml/src/stats.bend as Stats
import ./vendor/bend-ml/src/linear_regression.bend as Linear

def show(result: Result<&2, &2, Linear.FitError, Linear.Model>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.die(Unit, 1, Linear.error_message(error))
    case Done{model}:
      IO.print(F32.show(Linear.predict(model, 5.0)))

def main() -> IO(Unit):
  rows = Batch.from_list(Stats.Sample, [
    Stats.Sample{1.0, 3.0}, Stats.Sample{2.0, 5.0},
    Stats.Sample{3.0, 7.0}, Stats.Sample{4.0, 9.0}])
  show(Linear.fit(rows))
```

```sh
bend main.bend
# 11
```

Imports are relative to the importing file. There is no registry package to install: this experiment uses a Git checkout and local Bend imports. Pin that checkout to a commit when you need reproducibility.

## API

| Module | Public entry points |
| --- | --- |
| `src/batch.bend` | `Batch<A>`, `Empty`, `Item`, `Fork`, `from_list`, `to_list`, `count`, `map`, `fold` |
| `src/stats.bend` | `Sample{x, y}`, `Moments`, `moments`, `merge` |
| `src/linear_regression.bend` | `Model{slope, intercept}`, `FitError`, `fit`, `predict`, `predict_batch`, `error_message` |
| `src/metrics.bend` | `Prediction{actual, predicted}`, `mean_squared_error` |
| `src/numeric.bend` | `is_finite`, `squared_error`, `close(a, b, atol, rtol)` |

Other definitions in these modules are implementation helpers, even though Bend makes them importable.

`Sample{x, y}` contains one input feature and its target. `fit` returns `Done{Model{slope, intercept}}` or `Fail{FitError}`. Prediction evaluates `slope * x + intercept`.

Training errors:

- `NotEnoughSamples`: fewer than two observations.
- `NonFiniteData`: an input or target is NaN or infinite.
- `ConstantFeature`: input variance is zero at F32 precision.
- `NumericalFailure`: an intermediate statistic or coefficient is non-finite.

The minimum sample count is checked first. `predict` and `predict_batch` use ordinary F32 arithmetic without input validation; non-finite inputs or overflow can produce non-finite predictions.

`predict_batch(model, inputs)` takes `Batch<F32>` and returns predictions in the same tree shape and order. Use `Batch.to_list(F32, predictions)` to get a list. Declare `+model` or `+rows` when you need to reuse those values in Bend.

For metrics, create a `Batch<Metrics.Prediction>` of actual/predicted pairs and call `Metrics.mean_squared_error`. It returns `Some{value}`, or `None{}` for an empty batch or non-finite result. For example, pairs `(1, 2)` and `(3, 1)` have MSE `2.5`.

## CPU parallelism and an optional GPU experiment

The model merges centered statistics over independent subtrees using Bend's parallel call syntax. Batch prediction, mapping, and folding use the same structure.

Compile the small example to use native CPU execution (requires clang 14+):

```sh
mkdir -p build
bend examples/linear_regression.bend -o build/linear-regression
./build/linear-regression --threads 4
```

`examples/parallel.bend` generates 16,384 rows on the same line and calls `Linear.fit!`. Compile it to experiment with GPU execution:

```sh
bend examples/parallel.bend -o build/parallel
./build/parallel --threads 4
./build/parallel --gpu 1GB
```

The `!` build requires clang 19+ and the platform GPU toolchain: Metal on macOS, or CUDA 12 at `/usr/local/cuda` on Linux. Keep the generated `build/parallel.gpu` beside the executable. A GPU-capable binary can also run its work on the CPU without `--gpu`. Running this source directly with `bend` remains sequential JavaScript execution.

The 16,384-row example was also run successfully on native CPU and Metal GPU with Bend 2.0.5 on macOS. CUDA execution has not been tested.

These examples demonstrate execution modes, not speedups. Tiny datasets are likely dominated by setup costs. No comparison against scikit-learn or optimized numerical libraries has been made.

## Tests and proofs

```sh
./scripts/test.sh           # structural proofs, 19 tests, small example
./scripts/test.sh --native  # also compile and run tests on four CPU threads
```

The tests cover known coefficients, noisy data, a constant target, large input offsets, malformed training data, numerical overflow, metrics, and batch conversion/order. CI runs the same checks on Linux using the current Bend 2 release.

`LAWS.bend` declares two structural contracts, proved by `PROOF.bend`:

- Predicting a batch preserves its observation count.
- Empty training data returns `NotEnoughSamples`.

These are not proofs of numerical accuracy or model quality. Floating-point behavior is checked with numerical tests.

## Scope and limitations

- One feature and one target; no multiple regression, regularization, sample weights, missing-value handling, or model persistence yet.
- Arithmetic is F32. Centered statistics avoid the most obvious cancellation in raw sum-of-squares formulas, but rounding, underflow, ill-conditioned data, and overflow still matter. Rescale inputs when appropriate.
- This is intended for small experiments, well below 2^24 observations where F32 can no longer represent every integer count exactly.
- `Batch.from_list` builds a balanced tree in O(n log n) time. It uses linked lists and allocated tree nodes, not packed numerical arrays. Training traverses that tree in O(n) work; construction and allocation costs are separate.
- Constructing arbitrary `Fork` trees is supported, but strongly unbalanced trees reduce parallel efficiency. Floating-point results can differ with grouping.
- Bend is evolving quickly. Its language guide (`bend guide`) and compiler errors are useful when a newer release changes behavior.

## License

MIT. See [LICENSE](LICENSE).
