# Local benchmarks

Single-feature ordinary least squares with an intercept, using the public
`Linear.fit` implementation. The suite runs each implementation serially in a
fresh process, varies execution order with a recorded random seed, and keeps
all timing samples and coefficient checks.

## Published run

[Apple M2 Max, September 18, 2026](results/2026-09-18-m2-max.md) ·
[raw JSON](results/2026-09-18-m2-max.json). The run used the defaults below from
a clean commit; all sample counts and source hashes are recorded. This was a
daily-use workstation, not an isolated lab environment.

## Run on Apple Silicon

Prerequisites: Bend 2.0.5, clang 19+ with working Metal tooling, Node.js, and
Python 3.12. The Metal compiler must already work with Bend's GPU example.
Use [uv](https://docs.astral.sh/uv/) to install the pinned Python dependencies:

```sh
uv venv build/benchmark-venv --python 3.12
uv pip install --python build/benchmark-venv/bin/python -r benchmark/requirements.txt
build/benchmark-venv/bin/python -m unittest discover -s benchmark -p 'test_*.py'
build/benchmark-venv/bin/python benchmark/run.py
```

The default run compares 4,096, 65,536, and 1,048,576 rows, with 20 warmups and
15 recorded fits per process, repeated in three rounds. That is 45 measured
fits per implementation and dataset size. Compilation is done before any
measurements. Allow a few minutes and avoid other CPU/GPU-heavy work while it
runs. Results go to ignored `build/benchmark/results.json` and `results.md`.

For a quick check:

```sh
build/benchmark-venv/bin/python benchmark/run.py \
  --depths 12 --warmups 5 --trials 5 --rounds 1 \
  --output build/benchmark/quick.json
```

`--depths` selects powers of two (2 through 20). `--threads` controls the largest
Bend CPU run and the Python CPU library thread limits (default 12). The suite
also runs Bend with 1 and 4 threads. `--skip-mlx` omits MLX and `--skip-gpu` omits
GPU measurements; this driver still builds GPU-capable Bend binaries and needs
the platform GPU build toolchain. This suite is designed for Apple Silicon;
CUDA is not covered by the published results.

To save a run for publication, pass a new path such as
`--output benchmark/results/my-machine.json`. Review its metadata, commit the
JSON and generated Markdown, and link them from the root README. Prefer running
from a clean commit: each report records its code commit, source hashes, and
whether the working tree had changes.

## What is compared

| Implementation | Algorithm and representation |
| --- | --- |
| Bend CPU, 1 / 4 / 12 requested threads | Public `Linear.fit`, centered pairwise moments over a balanced `Batch` tree, F32 |
| Bend Metal GPU | The same fit via `!`, 1 GB GPU heap budget, F32 |
| scikit-learn `LinearRegression` | Dense least-squares estimator, float32 input; includes estimator creation and validation |
| NumPy | Centered two-pass formula with float32 means and BLAS dot products |
| SciPy `linregress` | Statistical regression API; float32 input, mixed float32/float64 intermediates and additional statistical outputs |
| MLX CPU / Metal | Compiled centered two-pass formula, float32; includes device completion and reading coefficients on the host |
| C / Node.js | Single-thread ports of Bend's pairwise centered-moment algorithm, float32 arithmetic over flat arrays |

These answer the same fitting problem, but have different algorithms, memory
layouts, validation, and API overhead. NumPy and MLX implement a specialized
one-feature formula, not a complete estimator API. The C and JavaScript controls
omit Bend's invalid-input checks. C uses `-O3 -ffp-contract=off` without fast-math;
JavaScript uses `Math.fround` for each arithmetic step. These results compare the
listed implementations; they are not a general language ranking.

Python CPU libraries receive the same thread limit as the largest Bend CPU run,
using `threadpoolctl` and `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS`. A limit is not a guarantee that
an operation uses that many threads. MLX manages its own CPU/GPU execution.
Bend CPU commands explicitly use `--gpu off`, because `!` otherwise automatically
uses an available GPU in Bend 2.0.5.

## Dataset and correctness

All implementations generate the same binary-exact float32 input values:

```text
x[i] = ((17*i mod 4096) - 2048) / 256
noise[i] = ((13*i mod 1024) - 512) / 8192
y[i] = 3*x[i] + 2 + noise[i]
```

The data repeats every 4,096 rows. Larger sizes measure additional work and memory
traffic, not richer data. This is synthetic, well-conditioned data, not a test of
numerical robustness, multiple features, sparse inputs, or real-world datasets.
The existing library tests exercise separate numerical edge cases.

The orchestrator computes a float64 reference from the same float32 arrays and
stores their SHA-256. Every warmup and measured fit is checked against an absolute coefficient tolerance of
**0.00001**. Finite results outside this tolerance are retained with an accuracy
flag in the table and raw data; they are not treated as equivalent-accuracy
speed comparisons. Missing samples, invalid durations, crashes, and non-finite
models abort the run. No outliers or failures are silently removed. Completed runs are saved after each
process so an interrupted run can be inspected, but incomplete reports have no
summary or completion timestamp.

## Timing boundaries

- The headline table is **warmed fit latency**, with the dataset already in its
  native representation. It excludes process startup, imports, compilation, data
  generation, tree/array construction, and warmups. It includes fit-time
  allocations, Bend reference management, and host-readable output coefficients.
- Setup is recorded once per process. Bend directly constructs a balanced tree;
  it does not call `Batch.from_list`. Setup timings across implementations are
  not identical operations. For example, MLX also creates/evaluates device arrays.
- First fits and all subsequent warmups are retained separately. Metal and MLX
  compilation/initialization may appear here, rather than in setup.
- Native drivers use monotonic clocks; Python uses `perf_counter_ns` and Node
  uses `performance.now`. Bend's custom foreign clock avoids the integer
  millisecond resolution of `IO.now`. Clocks return nanosecond units, but their
  effective resolution can be coarser.
- Bend's timing ends after matching the completed model. MLX explicitly calls
  `mx.eval` and reads both scalar outputs before ending the timer. GPU timings
  therefore include launch, synchronization, and returning the coefficients.
- Medians and p10–p90 ranges include all measured samples across three fresh
  processes. Per-round medians and maximum coefficient error are also retained.
  Percentiles describe this run's variation, not confidence intervals.
- Full process wall time is retained as context, including setup, warmups, fits,
  and output. It is not presented as a one-shot fit benchmark.

Sources for API behavior: [scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html),
[NumPy mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html),
[SciPy linregress](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html),
and [MLX compilation and synchronized timing](https://ml-explore.github.io/mlx/build/html/usage/compile.html).
