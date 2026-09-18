// Nanoseconds since the first call; stays within Bend's 48-bit Nat range.
static uint64_t bench_origin;
static Term bench_clock_run(Env e, Term* f, IoWork* w) {
  uint64_t now = io_tick();
  if (!bench_origin) bench_origin = now;
  return (Term)(now - bench_origin);
}
static void __attribute__((constructor)) bench_clock_use(void) {
  io_eff(CID_BENCH_CLOCK, bench_clock_run, 0);
}
