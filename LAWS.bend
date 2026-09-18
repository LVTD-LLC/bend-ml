import Base
import ./src/batch.bend as Batch
import ./src/linear_regression.bend as Linear

# Structural guarantees only: these make no claims about F32 accuracy.
law prediction_count:
  for +model: Linear.Model
  for +rows: Batch.Batch<F32>
  {Batch.count(F32, Linear.predict_batch(model, rows)) == Batch.count(F32, rows) : Nat}

law rejects_empty_training:
  {Linear.fit(Batch.Empty{}) == Fail{Linear.NotEnoughSamples{}} : Result<&2, &2, Linear.FitError, Linear.Model>}
