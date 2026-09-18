import Base
import ./batch.bend as Batch
import ./numeric.bend as Numeric

type Prediction is Data:
  Prediction{actual: F32, predicted: F32}

type SquaredErrors is Data:
  SquaredErrors{n: Nat, total: F32}

def singleton(row: Prediction) -> SquaredErrors:
  match row:
    case Prediction{actual, predicted}:
      SquaredErrors{1n, Numeric.squared_error(actual, predicted)}

def merge(a: SquaredErrors, b: SquaredErrors) -> SquaredErrors:
  match a b:
    case SquaredErrors{n1, s1} SquaredErrors{n2, s2}:
      SquaredErrors{Nat.add(n1, n2), F32.add(s1, s2)}

def checked(ok: Bool, value: F32) -> Maybe<&2, F32>:
  match ok:
    case False{}:
      None{}
    case True{}:
      Some{value}

def finish(stats: SquaredErrors) -> Maybe<&2, F32>:
  match stats:
    case SquaredErrors{n, total}:
      match n:
        case 0n:
          None{}
        case 1n+p:
          +value = F32.div(total, F32.from_nat(1n+p))
          checked(Numeric.is_finite(value), value)

# None means no observations or a non-finite numerical result.
def mean_squared_error(rows: Batch.Batch<Prediction>) -> Maybe<&2, F32>:
  finish(Batch.fold(~Prediction, ~SquaredErrors, ~singleton, ~merge, SquaredErrors{0n, 0.0}, rows))
