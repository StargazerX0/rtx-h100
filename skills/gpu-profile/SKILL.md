---
name: gpu-profile
description: Find where a CUDA workload's time goes before optimizing it. Kernel time split, the bound of each hot kernel, register and spill cost, and an Amdahl table of what each lever can buy. Use when a GPU job is slow or when choosing the next optimization.
---

# GPU profile before optimizing

1. **Split time per kernel** with nsys over the timed phase only; start the window at the first kernel of the timed work, not at load. Report share and time per unit of work (per sample, per token), so runs of different length compare.
2. **Name the bound of each hot kernel** with ncu (`--set full` on a few launches after warm-up): memory, local-memory traffic, divergence, L1/TEX or issue.
3. **Read static cost** from `ptxas -v`: registers, spill bytes, local memory per thread.
4. **Write an Amdahl table**: total gain if each kernel got 1.5x or 2x. It says which goals need a different algorithm rather than tuning.
5. **Predict, then measure.** Write the expected gain before each run. A prediction wrong in sign is a finding.

## Traps

- Results from another GPU do not transfer, even on the same die: bandwidth changes what spills cost.
- Shared GPUs give image checks, not timings.
- Parse profiler units per row; mixed us/ms in one column once invented a pathological kernel.
- Static instruction counts do not tell how often code runs; allocated memory is not touched memory.
- More registers can win when they remove spills; occupancy is not the whole story.
- Check for idle time between kernels before building a scheduler: in a wavefront path tracer it was under 6%.
