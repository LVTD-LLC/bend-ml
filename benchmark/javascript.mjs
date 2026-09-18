import { performance } from 'node:perf_hooks';
const [n, warmups, trials] = process.argv.slice(2).map(Number);
if (!Number.isInteger(n) || n < 2 || n > 2 ** 20 || !Number.isInteger(warmups) || warmups < 0 || !Number.isInteger(trials) || trials < 1) {
  throw new Error('Expected row count, warmup count, and trial count');
}
const f = Math.fround;
let start = performance.now();
const x = new Float32Array(n), y = new Float32Array(n);
for (let i = 0; i < n; i++) {
  x[i] = ((i * 17) % 4096 - 2048) / 256;
  y[i] = 3 * x[i] + 2 + ((i * 13) % 1024 - 512) / 8192;
}
console.log(JSON.stringify({ phase: 'setup', ns: Math.round((performance.now() - start) * 1e6) }));
// Same pairwise centered moments, rounding every arithmetic step to float32.
function moments(lo, hi) {
  if (hi - lo === 1) return [1, x[lo], y[lo], 0, 0];
  const mid = lo + Math.floor((hi - lo) / 2);
  const a = moments(lo, mid), b = moments(mid, hi);
  const count = a[0] + b[0], weight = f(b[0] / count), cross = f(a[0] * weight);
  const dx = f(b[1] - a[1]), dy = f(b[2] - a[2]);
  return [count, f(a[1] + f(dx * weight)), f(a[2] + f(dy * weight)),
    f(f(a[3] + b[3]) + f(f(dx * dx) * cross)),
    f(f(a[4] + b[4]) + f(f(dx * dy) * cross))];
}
for (let i = 0; i < warmups + trials; i++) {
  start = performance.now();
  const m = moments(0, n);
  const slope = f(m[4] / m[3]), intercept = f(m[2] - f(slope * m[1]));
  const ns = Math.round((performance.now() - start) * 1e6);
  console.log(JSON.stringify({ phase: i < warmups ? 'warmup' : 'fit', ns, slope, intercept }));
}
