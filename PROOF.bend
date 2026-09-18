import Base
import ./LAWS.bend as Laws
import ./src/batch.bend as Batch
import ./src/linear_regression.bend as Linear

def Laws.prediction_count(model, rows):
  match rows:
    case Batch.Empty{}:
      {==}
    case Batch.Item{value}:
      {==}
    case Batch.Fork{left, right}:
      %Laws.prediction_count(model, left) :
        {Nat.add(Batch.count(F32, Linear.predict_batch(model, left)), Batch.count(F32, Linear.predict_batch(model, right))) == Nat.add(_, Batch.count(F32, right)) : Nat}
      %Laws.prediction_count(model, right) :
        {Nat.add(Batch.count(F32, Linear.predict_batch(model, left)), Batch.count(F32, Linear.predict_batch(model, right))) == Nat.add(Batch.count(F32, Linear.predict_batch(model, left)), _) : Nat}
      {==}

def Laws.rejects_empty_training():
  {==}
