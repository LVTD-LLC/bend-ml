import Base
import ../src/batch.bend as Batch
import ../src/numeric.bend as Numeric
import ../src/stats.bend as Stats
import ../src/metrics.bend as Metrics
import ../src/linear_regression.bend as Linear

# Failures exit nonzero on both the JS and native backends.
def check(name: String, ok: Bool) -> IO(Unit):
  match ok:
    case False{}:
      IO.die(Unit, 1, "FAIL: " ++ name)
    case True{}:
      IO.print("PASS: " ++ name)

def near(a: F32, b: F32) -> Bool:
  Numeric.close(a, b, 0.0001, 0.00001)

def fitted(result: Result<&2, &2, Linear.FitError, Linear.Model>, slope: F32, intercept: F32) -> Bool:
  match result:
    case Fail{error}:
      False{}
    case Done{Linear.Model{s, i}}:
      near(s, slope) && near(i, intercept)

def fit(xs: List<&2, Stats.Sample>) -> Result<&2, &2, Linear.FitError, Linear.Model>:
  Linear.fit(Batch.from_list(Stats.Sample, xs))

def error_is(result: Result<&2, &2, Linear.FitError, Linear.Model>, expected: U32) -> Bool:
  match result:
    case Done{model}:
      False{}
    case Fail{error}:
      match error:
        case Linear.NotEnoughSamples{}:
          U32.is_eq(expected, 0)
        case Linear.NonFiniteData{}:
          U32.is_eq(expected, 1)
        case Linear.ConstantFeature{}:
          U32.is_eq(expected, 2)
        case Linear.NumericalFailure{}:
          U32.is_eq(expected, 3)

def metric_is(result: Maybe<&2, F32>, expected: F32) -> Bool:
  match result:
    case None{}:
      False{}
    case Some{value}:
      near(value, expected)

def metric_missing(result: Maybe<&2, F32>) -> Bool:
  match result:
    case None{}:
      True{}
    case Some{value}:
      False{}

def same_list(xs: List<&2, Nat>, ys: List<&2, Nat>) -> Bool:
  match xs ys:
    case Nil{} Nil{}:
      True{}
    case x <> xt y <> yt:
      Nat.is_eq(x, y) && same_list(xt, yt)
    case xs ys:
      False{}

# Test all sizes from zero to 33, including uneven split boundaries.
def roundtrips(+n: Nat) -> Bool:
  match n:
    case 0n:
      Nat.is_eq(Batch.count(Nat, Batch.from_list(Nat, [])), 0n)
    case 1n+p:
      +xs = List.range(1n+p)
      +rows = Batch.from_list(Nat, xs)
      Nat.is_eq(Batch.count(Nat, rows), 1n+p) &&
        same_list(Batch.to_list(Nat, rows), xs) && roundtrips(p)

def square(x: F32) -> F32:
  F32.square(x)

def same_floats(xs: List<&2, F32>, ys: List<&2, F32>) -> Bool:
  match xs ys:
    case Nil{} Nil{}:
      True{}
    case x <> xt y <> yt:
      near(x, y) && same_floats(xt, yt)
    case xs ys:
      False{}

def predictions(xs: List<&2, F32>) -> Bool:
  match xs:
    case a <> b <> c <> Nil{}:
      near(a, 1.0) && near(b, 5.0) && near(c, 11.0)
    case xs:
      False{}

def main() -> IO(Unit):
  do IO<Unit>:
    check("batch sizes and order 0..33", roundtrips(33n))
    check("batch map", predictions(Batch.to_list(F32,
      Linear.predict_batch(Linear.Model{2.0, 1.0}, Batch.from_list(F32, [0.0, 2.0, 5.0])))))
    check("shared map", same_floats(Batch.to_list(F32,
      Batch.map(~F32, ~F32, ~square, Batch.from_list(F32, [1.0, 2.0, 3.0]))), [1.0, 4.0, 9.0]))
    check("exact line", fitted(fit([
      Stats.Sample{1.0, 3.0}, Stats.Sample{2.0, 5.0},
      Stats.Sample{3.0, 7.0}, Stats.Sample{4.0, 9.0}]), 2.0, 1.0))
    check("two samples and negative slope", fitted(fit([
      Stats.Sample{0.0, 4.0}, Stats.Sample{2.0, 0.0}]), F32.sub(0.0, 2.0), 4.0))
    check("noisy least squares", fitted(fit([
      Stats.Sample{0.0, 1.0}, Stats.Sample{1.0, 2.0}, Stats.Sample{2.0, 2.0}]), 0.5, 1.1666667))
    check("constant target", fitted(fit([
      Stats.Sample{1.0, 7.0}, Stats.Sample{2.0, 7.0}, Stats.Sample{3.0, 7.0}]), 0.0, 7.0))
    check("centered statistics with large offset", fitted(fit([
      Stats.Sample{100000.0, 200001.0}, Stats.Sample{100001.0, 200003.0},
      Stats.Sample{100002.0, 200005.0}, Stats.Sample{100003.0, 200007.0}]), 2.0, 1.0))
    check("empty training rejected", error_is(fit([]), 0))
    check("one sample rejected", error_is(fit([Stats.Sample{1.0, 3.0}]), 0))
    check("constant feature rejected", error_is(fit([
      Stats.Sample{2.0, 3.0}, Stats.Sample{2.0, 5.0}]), 2))
    check("NaN input rejected", error_is(fit([
      Stats.Sample{F32.div(0.0, 0.0), 3.0}, Stats.Sample{2.0, 5.0}]), 1))
    check("infinite input rejected", error_is(fit([
      Stats.Sample{1.0, F32.div(1.0, 0.0)}, Stats.Sample{2.0, 5.0}]), 1))
    check("overflow rejected", error_is(fit([
      Stats.Sample{F32.sub(0.0, 1000000000000000000000000000000.0), 1.0}, Stats.Sample{1000000000000000000000000000000.0, 2.0}]), 3))
    check("mean squared error", metric_is(Metrics.mean_squared_error(Batch.from_list(Metrics.Prediction, [
      Metrics.Prediction{1.0, 2.0}, Metrics.Prediction{3.0, 1.0}])), 2.5))
    check("empty metric", metric_missing(Metrics.mean_squared_error(Batch.Empty{})))
    check("non-finite metric", metric_missing(Metrics.mean_squared_error(Batch.Item{
      Metrics.Prediction{F32.div(1.0, 0.0), 1.0}})))
    check("finite values", Numeric.is_finite(F32.sub(0.0, 4.5)) && Bool.not(Numeric.is_finite(F32.div(0.0, 0.0))) &&
      Bool.not(Numeric.is_finite(F32.div(1.0, 0.0))))
    check("single prediction", near(Linear.predict(Linear.Model{2.0, 1.0}, 10.0), 21.0))
    IO.print("All 19 tests passed.")
