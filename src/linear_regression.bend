import Base
import ./batch.bend as Batch
import ./numeric.bend as Numeric
import ./stats.bend as Stats

# Ordinary least squares with one input feature and a fitted intercept.
type Model is Data:
  Model{slope: F32, intercept: F32}

type FitError is Data:
  NotEnoughSamples{}
  NonFiniteData{}
  ConstantFeature{}
  NumericalFailure{}

def error_message(error: FitError) -> String:
  match error:
    case NotEnoughSamples{}:
      "Fit needs at least two samples."
    case NonFiniteData{}:
      "Training data contains NaN or infinity."
    case ConstantFeature{}:
      "Input feature has zero variance at F32 precision."
    case NumericalFailure{}:
      "F32 arithmetic overflowed or became invalid; rescale the data."

def checked_model(ok: Bool, slope: F32, intercept: F32) -> Result<&2, &2, FitError, Model>:
  match ok:
    case False{}:
      Fail{NumericalFailure{}}
    case True{}:
      Done{Model{slope, intercept}}

def solve(nonzero: Bool, mean_x: F32, mean_y: F32, xx: F32, xy: F32) -> Result<&2, &2, FitError, Model>:
  match nonzero:
    case False{}:
      Fail{ConstantFeature{}}
    case True{}:
      +slope = F32.div(xy, xx)
      +intercept = F32.sub(mean_y, F32.mul(slope, mean_x))
      checked_model(Numeric.is_finite(slope) && Numeric.is_finite(intercept), slope, intercept)

def checked_moments(ok: Bool, mean_x: F32, mean_y: F32, +xx: F32, xy: F32) -> Result<&2, &2, FitError, Model>:
  match ok:
    case False{}:
      Fail{NumericalFailure{}}
    case True{}:
      solve(F32.is_gt(xx, 0.0), mean_x, mean_y, xx, xy)

def checked_input(valid: Bool, +mean_x: F32, +mean_y: F32, +xx: F32, +xy: F32) -> Result<&2, &2, FitError, Model>:
  match valid:
    case False{}:
      Fail{NonFiniteData{}}
    case True{}:
      checked_moments(Numeric.is_finite(mean_x) && Numeric.is_finite(mean_y) &&
        Numeric.is_finite(xx) && Numeric.is_finite(xy), mean_x, mean_y, xx, xy)

def finish(stats: Stats.Moments) -> Result<&2, &2, FitError, Model>:
  match stats:
    case Stats.NoSamples{}:
      Fail{NotEnoughSamples{}}
    case Stats.Moments{n, mean_x, mean_y, xx, xy, valid}:
      match n:
        case 0n:
          Fail{NotEnoughSamples{}}
        case 1n+0n:
          Fail{NotEnoughSamples{}}
        case 1n+1n+rest:
          checked_input(valid, mean_x, mean_y, xx, xy)

def fit(rows: Batch.Batch<Stats.Sample>) -> Result<&2, &2, FitError, Model>:
  finish(Stats.moments(rows))

def predict(model: Model, x: F32) -> F32:
  match model:
    case Model{slope, intercept}:
      F32.add(F32.mul(slope, x), intercept)

def predict_batch(+model: Model, rows: Batch.Batch<F32>) -> Batch.Batch<F32>:
  match rows:
    case Batch.Empty{}:
      Batch.Empty{}
    case Batch.Item{x}:
      Batch.Item{predict(model, x)}
    case Batch.Fork{left, right}:
      a b = predict_batch(model, left) predict_batch(model, right)
      Batch.Fork{a, b}
