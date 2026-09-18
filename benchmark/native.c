#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Same centered pairwise algorithm as bend-ml, with flat input arrays and no
// heap allocations during fitting. Single thread; no fast-math or FMA fusion.
typedef struct { float n, x, y, xx, xy; } Moments;
static uint64_t now_ns(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
__attribute__((noinline))
static Moments moments(const float *x, const float *y, size_t lo, size_t hi) {
  if (hi - lo == 1) return (Moments){1, x[lo], y[lo], 0, 0};
  size_t mid = lo + (hi - lo) / 2;
  Moments a = moments(x, y, lo, mid), b = moments(x, y, mid, hi);
  float n = a.n + b.n, weight = b.n / n, cross = a.n * weight;
  float dx = b.x - a.x, dy = b.y - a.y;
  return (Moments){n, a.x + dx * weight, a.y + dy * weight,
    (a.xx + b.xx) + (dx * dx) * cross,
    (a.xy + b.xy) + (dx * dy) * cross};
}
int main(int argc, char **argv) {
  if (argc != 4) return 2;
  size_t n = strtoull(argv[1], NULL, 10);
  int warmups = atoi(argv[2]), trials = atoi(argv[3]);
  if (n < 2 || n > (1u << 20) || warmups < 0 || trials < 1) return 2;
  uint64_t begin = now_ns();
  float *x = malloc(n * sizeof(float)), *y = malloc(n * sizeof(float));
  if (!x || !y) return 1;
  for (size_t i = 0; i < n; ++i) {
    x[i] = ((float)((i * 17) % 4096) - 2048.0f) / 256.0f;
    float noise = ((float)((i * 13) % 1024) - 512.0f) / 8192.0f;
    y[i] = (3.0f * x[i] + 2.0f) + noise;
  }
  printf("{\"phase\":\"setup\",\"ns\":%llu}\n", (unsigned long long)(now_ns() - begin));
  for (int i = 0; i < warmups + trials; ++i) {
    begin = now_ns();
    // Prevent the optimizer from reusing a previous fit of unchanged arrays.
    __asm__ volatile("" : : "r"(x), "r"(y) : "memory");
    Moments m = moments(x, y, 0, n);
    float slope = m.xy / m.xx, intercept = m.y - slope * m.x;
    uint64_t elapsed = now_ns() - begin;
    printf("{\"phase\":\"%s\",\"ns\":%llu,\"slope\":%.9g,\"intercept\":%.9g}\n",
      i < warmups ? "warmup" : "fit", (unsigned long long)elapsed, slope, intercept);
  }
  free(x); free(y);
  return 0;
}
