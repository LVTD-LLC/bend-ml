import Base

# Reject both infinities and every NaN representation.
def is_finite(x: F32) -> Bool:
  U32.is_ne(U32.and(F32.bits(x), 2139095040), 2139095040)

def squared_error(actual: F32, predicted: F32) -> F32:
  F32.square(F32.sub(actual, predicted))

# Absolute + relative tolerance, useful when testing F32 results.
def close(+a: F32, +b: F32, atol: F32, rtol: F32) -> Bool:
  is_finite(a) && is_finite(b) &&
    F32.is_le(F32.abs(F32.sub(a, b)),
      F32.add(atol, F32.mul(rtol, F32.max(F32.abs(a), F32.abs(b)))))
