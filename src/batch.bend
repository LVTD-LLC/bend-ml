import Base

# A reusable tree of rows. Internal nodes expose independent work to Bend.
type Batch<-A: Data> is Data:
  Empty{}
  Item{value: A}
  Fork{left: Batch<A>, right: Batch<A>}

def count(-A: Data, rows: Batch<A>) -> Nat:
  match rows:
    case Empty{}:
      0n
    case Item{value}:
      1n
    case Fork{left, right}:
      a b = count(A, left) count(A, right)
      Nat.add(a, b)

# Split list ranges into a balanced tree; construction takes O(n log n).
# fuel bounds recursion; from_list supplies a sufficient bound (list length).
def build(-A: Data, +fuel: Nat, n: Nat, +xs: List<&2, A>) -> Batch<A>:
  match fuel n:
    case 0n n:
      Empty{}
    case 1n+p 0n:
      Empty{}
    case 1n+p 1n+0n:
      match xs:
        case Nil{}:
          Empty{}
        case h <> t:
          Item{h}
    case 1n+p 1n+1n+k:
      +size = Nat.add(2n, k)
      +half = Nat.div(size, 2n)
      left = build(A, p, half, List.take(&2, A, xs, half))
      right = build(A, p, Nat.sub(size, half), List.drop(&2, A, xs, half))
      Fork{left, right}

def from_list(-A: Data, +xs: List<&2, A>) -> Batch<A>:
  +n = List.length(&2, A, xs)
  build(A, n, n, xs)

def to_list(-A: Data, rows: Batch<A>) -> List<&2, A>:
  match rows:
    case Empty{}:
      Nil{}
    case Item{value}:
      [value]
    case Fork{left, right}:
      List.append(&2, A, to_list(A, left), to_list(A, right))

# Template callbacks are reusable and specialized at compile time.
def map(~A: Data, ~B: Data, ~f: A -> B, rows: Batch<A>) -> Batch<B>:
  match rows:
    case Empty{}:
      Empty{}
    case Item{value}:
      Item{f(value)}
    case Fork{left, right}:
      a b = map(~A, ~B, ~f, left) map(~A, ~B, ~f, right)
      Fork{a, b}

# The join operation should have an identity zero and be associative in the
# intended arithmetic. F32 results can vary with grouping because of rounding.
def fold(~A: Data, ~B: Data, ~leaf: A -> B, ~join: B -> B -> B, +zero: B, rows: Batch<A>) -> B:
  match rows:
    case Empty{}:
      zero
    case Item{value}:
      leaf(value)
    case Fork{left, right}:
      a b = fold(~A, ~B, ~leaf, ~join, zero, left) fold(~A, ~B, ~leaf, ~join, zero, right)
      join(a, b)
