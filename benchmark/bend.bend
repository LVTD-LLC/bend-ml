import Base
import ../src/batch.bend as Batch
import ../src/stats.bend as Stats
import ../src/linear_regression.bend as Linear

def bench_clock() -> IO(Nat):
  import "./clock.c"
  import "./clock.js"

# The same binary-exact float32 dataset as the Python, C, and JS drivers.
def rows(+depth: Nat, +start: U32) -> Batch.Batch<Stats.Sample>:
  match depth:
    case 0n:
      +x = F32.div(F32.sub(U32.to_f32(U32.mod(U32.mul(start, 17), 4096)), 2048.0), 256.0)
      noise = F32.div(F32.sub(U32.to_f32(U32.mod(U32.mul(start, 13), 1024)), 512.0), 8192.0)
      Batch.Item{Stats.Sample{x, F32.add(F32.add(F32.mul(3.0, x), 2.0), noise)}}
    case 1n+p:
      a b = rows(p, start) rows(p, U32.add(start, U32.shln(1, p)))
      Batch.Fork{a, b}

def fitted(result: Result<&2, &2, Linear.FitError, Linear.Model>, start: Nat, phase: String) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.die(Unit, 1, Linear.error_message(error))
    case Done{Linear.Model{slope, intercept}}:
      do IO<Unit>:
        end : Nat <- bench_clock()
        IO.print("{\"phase\":\"" ++ phase ++ "\",\"ns\":" ++ Nat.show(Nat.sub(end, start)) ++
          ",\"slope\":" ++ F32.show(slope) ++ ",\"intercept\":" ++ F32.show(intercept) ++ "}")

def samples(n: Nat, +data: Batch.Batch<Stats.Sample>, +phase: String) -> IO(Unit):
  match n:
    case 0n:
      IO.pure(Unit, Unit{})
    case 1n+p:
      do IO<Unit>:
        start : Nat <- bench_clock()
        fitted(Linear.fit!(data), start, phase)
        samples(p, data, phase)

def prepared(+data: Batch.Batch<Stats.Sample>, start: Nat, warmups: Nat, trials: Nat) -> IO(Unit):
  do IO<Unit>:
    end : Nat <- bench_clock()
    IO.print("{\"phase\":\"setup\",\"ns\":" ++ Nat.show(Nat.sub(end, start)) ++ "}")
    samples(warmups, data, "warmup")
    samples(trials, data, "fit")

def run(depth: Nat, warmups: Nat, trials: Nat) -> IO(Unit):
  do IO<Unit>:
    start : Nat <- bench_clock()
    prepared(rows(depth, 0), start, warmups, trials)
