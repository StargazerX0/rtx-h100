---
name: kernel-specialize
description: Make exact, output-identical CUDA kernels specialized to known inputs (fixed scenes, models or shapes). Launch bounds, feature masks, dead-code pruning, compile-time constants, inlining, per-input binaries, each behind a macro. Use when a benchmark or deployment fixes its inputs and speed must come without changing results.
---

# Per-input kernel specialization

Order that paid on a path tracer (2.0x, output identical):

1. **Launch bounds per kernel and input**: sweep block size and min blocks; read registers and spills.
2. **Compile only the features the input uses** (feature masks, `#if` on capabilities).
3. **More work in flight** when the scheduler allows it; keep raising it until the measurement stops paying.
4. **Prune interpreters and dispatch**: log which opcodes or node types the input uses, compile the rest out, add a host check that aborts on an unexpected one. The smaller code lets the compiler inline hot paths.
5. **Fold constant parameters** into the binary as compile-time constants from a dump of the real run.
6. **Straight-line code for the hottest cases** instead of interpretation.

## Rules

- Every change behind a macro or a variable. With it off, the binary must be byte-identical to before (sha256). With it on, compare the output with the reference against the seed-to-seed noise floor, e.g. `scripts/identity.py`; FP-level changes sit at 3e-5 to 3e-3 of the floor.
- Load per-input binaries at run time (an env var naming the cubin), so one host serves all inputs.
- Derive choices from input content where possible; on the exact build a rule-based choice matched hand tuning.

## Traps

- **Folded constants freeze settings.** A binary with compile-time parameters ignores later runtime changes; sweeps on it measure nothing. Sweep on unfolded builds.
- Diff the full flag lists before comparing two builds; a missing define silently changes code.
- A failed compile blamed on memory was a real compile error; read the first error line before changing the build.
