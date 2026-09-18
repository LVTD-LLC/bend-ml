# Working on bend-ml

- Read `bend guide` and inspect `bend base` before changing Bend code. This is Bend 2, not the legacy HVM-based language.
- Keep the experiment small: shared primitives and one-feature ordinary least squares. Add abstractions only when an implementation needs them.
- Run `./scripts/test.sh --native` before committing. It checks `PROOF.bend`, numerical tests on JS and native CPU, and the README example.
- Keep structural contracts in `LAWS.bend` and their implementations in `PROOF.bend`. Do not weaken an existing law to accommodate a code change.
- Floating-point behavior is covered by numerical tests, not formal proof claims. Do not claim accuracy or performance beyond measured evidence.
- Preserve the balanced fork/join structure in batch reductions and predictions. Avoid `@unsafe`.
- Keep README install commands, imports, examples, and limitations aligned with the tested code. Generated binaries belong in ignored `build/`.
