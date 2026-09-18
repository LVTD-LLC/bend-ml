const bench_origin = performance.now();
function bench_clock() {
  return BigInt(Math.round((performance.now() - bench_origin) * 1e6));
}
