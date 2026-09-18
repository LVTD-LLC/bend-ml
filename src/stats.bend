import Base
import ./batch.bend as Batch
import ./numeric.bend as Numeric

type Sample is Data:
  Sample{x: F32, y: F32}

# Centered sufficient statistics, combined with the parallel variance formula.
# valid tracks non-finite input separately from arithmetic overflow.
type Moments is Data:
  NoSamples{}
  Moments{n: Nat, mean_x: F32, mean_y: F32, xx: F32, xy: F32, valid: Bool}

def singleton(row: Sample) -> Moments:
  match row:
    case Sample{+x, +y}:
      Moments{1n, x, y, 0.0, 0.0, Numeric.is_finite(x) && Numeric.is_finite(y)}

def merge(a: Moments, b: Moments) -> Moments:
  match a b:
    case NoSamples{} b:
      b
    case a NoSamples{}:
      a
    case Moments{+n1, +x1, +y1, xx1, xy1, v1} Moments{+n2, x2, y2, xx2, xy2, v2}:
      +n = Nat.add(n1, n2)
      +weight = F32.div(F32.from_nat(n2), F32.from_nat(n))
      +cross_weight = F32.mul(F32.from_nat(n1), weight)
      +dx = F32.sub(x2, x1)
      +dy = F32.sub(y2, y1)
      Moments{n,
        F32.add(x1, F32.mul(dx, weight)),
        F32.add(y1, F32.mul(dy, weight)),
        F32.add(F32.add(xx1, xx2), F32.mul(F32.square(dx), cross_weight)),
        F32.add(F32.add(xy1, xy2), F32.mul(F32.mul(dx, dy), cross_weight)),
        v1 && v2}

def moments(rows: Batch.Batch<Sample>) -> Moments:
  Batch.fold(~Sample, ~Moments, ~singleton, ~merge, NoSamples{}, rows)
