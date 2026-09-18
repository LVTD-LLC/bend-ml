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
