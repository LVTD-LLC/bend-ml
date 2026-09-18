# bend-ml

A machine-learning library for [Bend 2](https://bend-lang.com/).

Available on [BendHub](https://hub.bend-lang.com/0x53dde92df75268075f1bc6c33ecbb079).

One model, a few shared building blocks, and examples you can change:

- **Linear regression:** ordinary least squares for one feature, with a fitted intercept.
- **Batches:** reusable trees, list conversion, mapping, and parallel reductions.
- **Statistics:** mergeable centered moments for training.
- **Metrics:** mean squared error and floating-point comparison helpers.

Tested with **Bend 2.0.5**. The library has no external dependencies beyond Bend's `Base`. It does not use legacy Bend 1 / HVM.

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

## Use a Git checkout in your Bend project

From your project's root, place the library in `vendor/`:

```sh
mkdir -p vendor
git clone https://github.com/lvtd-llc/bend-ml.git vendor/bend_ml
```

Save this as `main.bend` in your project root:

```python
import Base
import ./vendor/bend_ml/src/batch.bend as Batch
import ./vendor/bend_ml/src/stats.bend as Stats
import ./vendor/bend_ml/src/linear_regression.bend as Linear

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

Use the underscore in `vendor/bend_ml`: Bend 2.0.5 can generate invalid JavaScript for imported paths containing a hyphen. Imports are relative to the importing file. These instructions use a Git checkout and local Bend imports; [BendHub imports](#use-from-bendhub) are also available. Pin that checkout to a commit when you need reproducibility.

## Use from BendHub

No Git checkout or separate package installation is required. Save the following
as `main.bend` and run `bend main.bend`; it prints `11`.

```bend
import Base
import 0x53dde92df75268075f1bc6c33ecbb079/src/batch.bend as Batch
import 0x53dde92df75268075f1bc6c33ecbb079/src/stats.bend as Stats
import 0x53dde92df75268075f1bc6c33ecbb079/src/linear_regression.bend as Linear

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

Bend downloads the package into `~/.bend/lib` on the first run, verifies its
content hashes, and uses the cached files on later runs. The first run requires
network access. The hash pins this exact release; BendHub currently has no named
packages or version numbers.

The [published package](https://hub.bend-lang.com/0x53dde92df75268075f1bc6c33ecbb079)
contains all five `src/` modules, `LAWS.bend`, `PROOF.bend`, and the
[`bend_ml.bend`](bend_ml.bend) publishing entry point, including the MIT license.
Import the individual API modules as above. The entry point collects and checks
the modules and proofs; it does not re-export their APIs.

To verify the published example from this repository:

```sh
bend examples/from_hub.bend
```

To publish a new release after making and testing changes:

```sh
./scripts/test.sh --native
bend bend_ml.bend --publish
```

The publisher uploads the entry point and its imported source files. It does not
upload benchmark results, generated binaries, or unrelated repository files.
Changed package contents produce a new hash; update the documented imports and
`examples/from_hub.bend` after verifying that new release.

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

## CPU and GPU execution

The model merges centered statistics over independent subtrees using Bend's parallel call syntax. Batch prediction, mapping, and folding use the same structure.

Compile the small example to use native CPU execution (requires clang 14+):

```sh
mkdir -p build
bend examples/linear_regression.bend -o build/linear-regression
./build/linear-regression --threads 4
```

`examples/parallel.bend` generates 16,384 rows on the same line and calls `Linear.fit!`. Compile it for native CPU or GPU execution:

```sh
bend examples/parallel.bend -o build/parallel
./build/parallel --threads 4 --gpu off
./build/parallel --gpu 1GB
```

The `!` build requires clang 19+ and the platform GPU toolchain: Metal on macOS, or CUDA 12 at `/usr/local/cuda` on Linux. Keep the generated `build/parallel.gpu` beside the executable. Bend 2.0.5 automatically uses an available GPU for `!` calls; pass `--gpu off` to force CPU execution. Running this source directly with `bend` remains sequential JavaScript execution.

The 16,384-row example was also run successfully on native CPU and Metal GPU with Bend 2.0.5 on macOS. CUDA execution has not been tested.

For measured comparisons against other implementations, see [Benchmarks](#benchmarks).

## Benchmarks

Measured locally on **Apple M2 Max: 12 CPU cores (8 performance + 4 efficiency), 38 GPU cores, 32 GB memory**, on AC power, macOS 26.5.2, September 18, 2026. Bend version: **2.0.5**.

The workload is single-feature linear regression with an intercept on identical, synthetic float32 data. Times below are **median warmed fit latency in milliseconds**; lower is better. Each cell contains 45 measured fits across three fresh processes, after 20 warmups per process. Execution order was randomized with a fixed seed.

| Implementation | 4,096 rows | 65,536 rows | 1,048,576 rows |
| --- | ---: | ---: | ---: |
| Bend CPU, 1 thread | 0.105 | 0.867 | 13.345 |
| Bend CPU, 4 threads | 0.128 | 0.276 | 3.561 |
| Bend CPU, 12 threads | 0.233 | 0.303 | 2.087 |
| Bend Metal GPU | 0.735 | 0.860 | 1.907 |
| scikit-learn LinearRegression (F32) | 0.229 | 0.834 | 8.337 |
| NumPy centered OLS (F32) | 0.012 | 0.050 | 0.666 |
| SciPy linregress (mixed F32/F64) | 0.164 | 0.221 | 1.091 |
| MLX compiled OLS, CPU (F32) | 0.034 | 0.079 | 0.963 † |
| MLX compiled OLS, Metal (F32) | 0.258 | 0.297 | 0.483 |
| C pairwise OLS, 1 thread (F32) | 0.027 | 0.422 | 6.971 |
| Node.js pairwise OLS, 1 thread (F32) | 0.118 | 1.695 | 23.351 |

**What this run shows:** at one million rows, Bend on 12 CPU threads took about **one-quarter of scikit-learn’s fit time**. Bend CPU and Metal timings were similar; their p10–p90 ranges overlap. The specialized NumPy and MLX Metal formulas were faster than Bend. At 4,096 rows, Bend’s single-thread path beat its parallel paths.

**Timing boundaries matter:** the table excludes data/tree construction, process startup, imports, compilation, and warmups. For example, constructing the one-million-row Bend GPU dataset took a median **33.951 ms** before fitting. These are repeated-fit measurements on reused data, not one-shot application timings. GPU fits include synchronization and reading the coefficients.

These implementations differ in algorithms, data layouts, and API overhead. scikit-learn provides a general estimator with input validation; NumPy and MLX use a specialized one-feature formula. C and Node.js port the pairwise algorithm over flat arrays on one CPU thread. Python CPU libraries were allowed up to 12 threads; actual utilization depends on the operation. The synthetic data repeats every 4,096 rows, so these results do not establish performance on other ML workloads.

† MLX CPU at one million rows had a maximum coefficient error of **0.0000572**, exceeding this benchmark’s absolute **0.00001** tolerance. Its timing is retained for transparency, not treated as an equivalent-accuracy comparison. Every other entry passed. Floating-point errors and all individual samples are recorded.

**Reproduce and inspect:** [benchmark instructions and methodology](benchmark/README.md), [full table with p10–p90 ranges](benchmark/results/2026-09-18-m2-max.md), and [raw samples, environment, source hashes, and correctness checks](benchmark/results/2026-09-18-m2-max.json).

```sh
uv venv build/benchmark-venv --python 3.12
uv pip install --python build/benchmark-venv/bin/python -r benchmark/requirements.txt
build/benchmark-venv/bin/python benchmark/run.py
```

## Tests and proofs

```sh
./scripts/test.sh           # proofs, 19 tests, local, vendored, and BendHub examples
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
- Keep datasets well below 2^24 observations where F32 can no longer represent every integer count exactly.
- `Batch.from_list` builds a balanced tree in O(n log n) time. It uses linked lists and allocated tree nodes, not packed numerical arrays. Training traverses that tree in O(n) work; construction and allocation costs are separate.
- Constructing arbitrary `Fork` trees is supported, but strongly unbalanced trees reduce parallel efficiency. Floating-point results can differ with grouping.
- Bend is evolving quickly. Its language guide (`bend guide`) and compiler errors are useful when a newer release changes behavior.

## License

MIT. See [LICENSE](LICENSE).
