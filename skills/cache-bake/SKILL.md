---
name: cache-bake
description: Precompute a per-input neural cache offline (hash-grid encoding plus small MLP, tiny-cuda-nn style), calibrate it, freeze it and use it to cut work at run time; choose where to query it; and avoid the failures of online training. Use when inputs are known in advance and a learned lookup can replace repeated computation, e.g. radiance caching in a renderer.
---

# Baking a neural cache

A frozen per-scene radiance cache that ended every path at its first hit gave +82% on a tuned path tracer and passed a PSNR gate.

1. **Train offline on unbiased targets**: generate fresh records from exact runs, use each record once, lower the learning rate in stages.
2. **Calibrate** one output scale on held-out seeds against an exact run.
3. **Freeze** for inference; use a fused kernel for the MLP so the query costs a small share of the frame.
4. **Choose the query point** by speed vs margin: earliest query is fastest and most biased.
5. **Gate** it (`quality-gate`) on all inputs and several seeds, and measure its quality against render time: a cache's bias caps quality, so past some sample count the exact path is faster.

Recipe example: `nrc/bake/bake.sh`, `nrc/bake/calibrate.sh`, `nrc/configs/t24.json`.

## Traps

- **Online training during the run left a smooth bias** that no setting removed; a plain per-cell mean beat it under a strict gate.
- **Capacity does not fix bias**: a 32x larger table and a 2x wider MLP left the same bias map. ReLU beat SiLU at every width.
- **Plain L2 loss.** With relative L2 the cache drifted 2-3x high and the network died. Too high a learning rate killed every ReLU. The bake that shipped used Adam beta2 0.9999; a separate MLP screen preferred 0.99, so tune it.
- **Sample records per vertex, not by throughput**, or deep queries read low.
- **Match run-time inputs to bake-time inputs** (limits, budgets, settings), or pin them.
- Check input precision: fp16 positions in a large scene box quantize to centimeters.
