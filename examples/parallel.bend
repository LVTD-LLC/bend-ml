import Base
import ../src/batch.bend as Batch
import ../src/stats.bend as Stats
import ../src/linear_regression.bend as Linear
import ./linear_regression.bend as Example

# Generate 16,384 observations on y = 2x + 1 as an already balanced tree.
def rows(+depth: Nat, +start: U32) -> Batch.Batch<Stats.Sample>:
  match depth:
    case 0n:
      +x = U32.to_f32(start)
      Batch.Item{Stats.Sample{x, F32.add(F32.mul(2.0, x), 1.0)}}
    case 1n+p:
      a b = rows(p, start) rows(p, U32.add(start, U32.shln(1, p)))
      Batch.Fork{a, b}

def main() -> IO(Unit):
  # ! permits GPU execution when the native binary is run with --gpu.
  Example.report(Linear.fit!(rows(14n, 0)))
