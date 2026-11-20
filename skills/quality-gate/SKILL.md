---
name: quality-gate
description: Accept or reject a speedup that changes the output (sampling shortcuts, caches, quantization, early termination) with a gate fixed in advance against a high-quality reference, and check that the speedup is real per unit of output quality, not only per sample or per token. Use before reporting any score of an approximate configuration.
---

# Gating an approximation

**Gate used for renders** (`scripts/gate.py`): per input, PSNR of the approximate output against a reference at 16x its cost must be at least the mean PSNR of a cheap exact baseline against the same reference, and the mean signal (luminance) may shift by at most 0.5%. Any domain needs the same three parts: a reference far better than both sides, a bar set by an exact baseline, and a bias limit.

1. Fix and date the gate before the first approximate run.
2. Check every reported arm on every input; screens without a check are not results.
3. Accept on seeds the tuning never saw, and check several seeds for the reported arm.
4. Measure quality against time, not only throughput: render the approximate and the exact configuration at several sample counts and compare the time each needs to reach the same quality.
5. When the gate changes, re-judge earlier arms and state which gate each number used.

## Traps

- **Throughput is not quality per second.** A cached sample did far less work than an exact one: the cache more than tripled samples per minute, yet its quality levelled off near 33 dB, and above about 1,000 samples the exact build reached the same quality sooner. The gate's own operating point sat past that crossover.
- **Bias binds before noise.** Most failures were a mean shift, not a PSNR loss, and a biased method's mean also wanders more at low sample counts.
- **Tuning on the acceptance seeds overfits**: +1.5% found on one seed pair failed 8 of 9 others.
- **Every metric has a blind spot.** A perceptual metric against one noisy sample punished a smooth 10% bias; a PSNR bar set by a noisy baseline accepts it. The best passing score doubled when the gate changed. Report the gate with the number.
- **Calibration hides bias**: a global scale fitted to an exact run makes the mean-shift test pass by construction. Fit on seeds the gate does not use.
- **Uncounted precompute.** Offline work the timed run relies on (a bake, a cache, a compile) is part of the cost; state it next to the score.
