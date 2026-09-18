import Base
import ../src/batch.bend as Batch
import ../src/stats.bend as Stats
import ../src/linear_regression.bend as Linear

# Four observations on y = 2x + 1.
def training_data() -> Batch.Batch<Stats.Sample>:
  Batch.from_list(Stats.Sample, [
    Stats.Sample{1.0, 3.0}, Stats.Sample{2.0, 5.0},
    Stats.Sample{3.0, 7.0}, Stats.Sample{4.0, 9.0}])

def report(result: Result<&2, &2, Linear.FitError, Linear.Model>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.die(Unit, 1, Linear.error_message(error))
    case Done{+model}:
      do IO<Unit>:
        IO.print("Fitted y = 2x + 1")
        IO.print("Prediction at x=5 (expected 11): " ++ F32.show(Linear.predict(model, 5.0)))
        IO.print("Prediction at x=10 (expected 21): " ++ F32.show(Linear.predict(model, 10.0)))

def main() -> IO(Unit):
  report(Linear.fit(training_data()))
