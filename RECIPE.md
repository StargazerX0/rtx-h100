# Recipe: 48,641 samples/min on one H100

On one H100, Blender 5.3.0 Cycles renders the Open Data benchmark at 48,641 samples per minute, 6.9 times stock.
Exact per-scene kernel specialization gives the first 2.0 times. A per-scene baked neural radiance cache that ends every path at its first hit gives most of the rest.
The `nrc` configuration passes the image gate at seed 0, the only seed its monster and junkshop cubins can render; the same cache on an earlier build misses the luminance limit on monster at seed 2.
Wide BVHs, kernel concurrency and online cache training gave nothing.

## 1. Target

- Hardware: one H100 SXM 80 GB in an 8-GPU node, CUDA 12.9. Cycles' CUDA backend uses no RT cores.
- Benchmark: Blender Open Data, launcher 3.3.0, benchmark script 3.1.2, scenes monster, junkshop and classroom.
- Score: samples per minute per scene, summed over the three scenes.
- Official protocol: per scene, a warm-up process, then a fresh timed process that renders for 30 s. `scripts/direct_bench.py` runs Blender with the launcher's own arguments, because the launcher needs opendata.blender.org even to run. A configuration's score is the sum of the per-scene medians over 6 runs.
- Screening protocol: three renders in one process after one scene load (`BENCH_REPEATS=3`). It matched the official protocol within 0.2% on the same configuration. Use it to screen; report official numbers only.
- Rules: each scene may be overfit on its own. Per-scene cubins, settings, hard-coded shader ids, materials compiled to native code and baked caches are all allowed. Nothing has to generalize to other scenes.

## 2. The image gate

An exact change must render the stock image up to floating-point rounding. An approximate change must pass this gate on every scene:

1. PSNR of the 4096-spp image at seed 0 against a 65,536-spp exact reference is at least the mean PSNR of exact 32-spp renders against the same reference.
2. Mean luminance differs from the reference by at most 0.5%.

The reference is the mean of four exact 16,384-spp renders at seeds 100-103. The bar is the mean over eight exact 32-spp renders at seeds 200-207. PSNR is computed on display-mapped images: `y = xk/(1 + xk)` with `k = 0.5 / median reference luminance`, then gamma 1/2.2. Checker: `scripts/gate.py`.

| scene | 32-spp bar | `nrc` | luminance |
|---|---|---|---|
| monster | 24.25 dB | 33.05 dB | -0.11% |
| junkshop | 18.84 dB | 23.80 dB | +0.23% |
| classroom | 23.84 dB | 32.31 dB | -0.10% |

The gate changed several times on the last day. From late morning until 19:00 ET it was HDR-FLIP against exact seed 1, at most the seed-to-seed floor plus 0.02, with the same luminance limit. For twenty minutes it was PSNR against exact seed 1, at most 5 dB under the seed-to-seed floor. Then it became the gate above. Under FLIP the baked cache passed only with a conservative rule, +12.5% on the exact build; rules c0 and d1 failed by 3 to 7 times the cap, and the best official scores were about 23,000. The 32-spp gate accepts a smooth bias of about ±10% at object scale that FLIP rejected. Read the final number with that in mind.

## 3. Build

Run the commands from the repository root. `$W` is one work directory; every later step and `slurm/repro.sbatch` expect this layout in it. The pinned source is Blender tag v5.3.0, commit `02f0899d78e374b63e7181d9061e1f7ab03419cc`.

```bash
scripts/fetch.sh $W                           # blender/ blender-5.3.0-linux-x64/ bench/ bakes/
scripts/build_host.sh $W/blender $W/host      # Blender + patches/host.diff -> host/install
scripts/build_engine.sh $W/engine             # tiny-cuda-nn + nrc/ -> engine/libcycles_nrc.so

scripts/kernel_tree.sh $W/blender $W/tree_nrc        # intern/cycles + patches/kernel/01..10
scripts/build_cubins.sh $W/tree_nrc/intern/cycles $W/nrc_cubins
(cd $W/nrc_cubins && sha256sum -c "$OLDPWD/configs/nrc_cubins.sha256")       # OK three times

scripts/kernel_tree.sh $W/blender $W/tree_exact 03   # intern/cycles + patches/kernel/01..03
FLAGS=exact_nvcc_flags.txt scripts/build_cubins.sh $W/tree_exact/intern/cycles $W/exact_cubins
(cd $W/exact_cubins && sha256sum -c "$OLDPWD/configs/exact_cubins.sha256")   # OK three times
```

- Needs: an H100; git, git-lfs, curl and rsync; the CUDA 12.9 toolkit (`NVCC` and `CUDA_HOME` override its path, default `/usr/local/cuda-12.9`); GCC 14 or newer (`GCC_TOOLSET_DIR`, default `/opt/rh/gcc-toolset-14/root/usr`); CMake; Python 3 with numpy and OpenEXR for the checks. Building takes about 30 GB of disk and two hours on 16 cores, most of it the host.
- The kernel and the host are separate trees, as in the project: the host never compiles CUDA, and each scene loads its own cubin through `CYCLES_CUDA_CUBIN`.
- `configs/<scene>/nrc_nvcc_flags.txt` holds every compile flag of the `nrc` cubins. `svm_nodes_used.h` lists the shader node types the scene uses; the interpreter cases of every other type compile out. `kdata_const.h` folds the scene's KernelData into the cubin as compile-time constants.
- The cubins are byte-identical to the ones the 48,641 run used. Check: `configs/nrc_cubins.sha256`.
- The `exact` configuration, 14,106 with images equal to stock up to floating-point rounding, uses the same host and the tree after patch 03.
- The per-scene headers ship generated. To regenerate them: `scripts/probe_nodes.sh` and `scripts/svm_nodes_header.py` give `svm_nodes_used.h`; cubins built with `-DCYCLES_KDATA_DUMP` instead of `-DCYCLES_KDATA_CONST`, run once with `BENCH_LOG=<file> scripts/run.sh nrc bench` and once with `run.sh nrc check`, give the dumps that `scripts/gen_kconst.py` folds into `kdata_const.h`.
- Comments in the patches and in `nrc/` cite the project's private notes: `analysis/*.md`, `jobs/`, `results/`, tree names such as `tree53-merge7`, and experiment numbers such as E46. Those files are not part of this repository; RECIPE.md carries what they established.
- Every change sits behind a macro or an environment variable. With all of them off, the host behaves as stock and each cubin is byte-identical to stock's with the same flags.

## 4. The steps

One kernel patch per milestone. `scripts/kernel_tree.sh <checkout> <out> NN` stops after patch NN. This repository holds the compile flags and run settings of the `exact` and `nrc` configurations only.

| patch | milestone | score | gain | image | what changed |
|---|---|---|---|---|---|
| | stock 5.3.0 | 7,041 | | stock | official binary, sm_90 cubin from stock source |
| 01-lb | lb2 | 8,810 | +25.1% | exact | per-kernel launch bounds: intersect 64x16, shade 512x2 |
| 01-lb | b53a | 9,812 | +11.4% | exact | per-scene feature masks, FMA slab test, BVH2 leaf size 2, spatial splits, shade 1024x1 on junkshop and classroom |
| 01-lb | 8x states | 10,675 | +8.8% | exact | 8 times the concurrent path states |
| 02-df | df2p | 11,168 | +4.6% | exact | dynamic-fetch BVH2 traversal for closest and shadow rays; 131,072-path sort partitions |
| 03-best | bhotB | **14,106** | +26.3% | exact | SVM node pruning per scene, closure array sized per scene, forced inlining of hot functions, dynamic-fetch SSS walk on monster |
| 04-merge1 | approxD | 20,654 | +46.4% | FLIP | passes off, light threshold 0.2, scrambling distance 0.0625, Russian roulette p = t, on classroom t^0.75; SSS walk roulette; queue bookkeeping |
| 05-merge2 | approxG0 | 21,623 | +4.7% | FLIP | BVH2 treelet optimization and 2x shadow pool, both exact; roulette p = t^1.5, clamp-aware on classroom; glossy bounces at most 2 |
| 06-merge4 | approxG | 22,050 | +2.0% | FLIP | approximate bump and texture lookups on indirect bounces; monster walk roulette 0.95; junkshop BVH4 shadow rays |
| 07-merge5 | approxH | 22,760 | +3.2% | FLIP | Sobol table cap 1024 on monster; per-scene scrambling and light threshold; 16x states |
| 08-merge7 | approxI | 22,960 | +0.9% | FLIP | indirect-only SSS walk roulette on monster; Sobol cap 256, light threshold 0.35 and 24x states on classroom; lower scrambling distance on junkshop and classroom |
| 09-mergeN | approxN | 27,041 | +17.8% | 42.9 / 39.7 / 42.5 dB | compile-time KernelData +2.1%; flat leaf triangles +1.2%; shadow child order; per-material shading programs +4.3%; BVH2 SAH tuning; a per-cell mean radiance cache that ends paths +7.0% |
| 10-final | **approxFinal = `nrc`** | **48,641** | +79.9% | 33.1 / 23.8 / 32.3 dB | the baked neural cache on rule c0 replaces the per-cell cache; emitter bake +5.7%; monster SSS local-intersection fix |

Notes on the table:

- Each gain is the row's score over the previous row's. The scores are official medians from different jobs and nodes; node spread is about 0.3%. Per-component gains inside a row come from single-node runs and do not add up: under rule c0 most deep-bounce gains overlap.
- The baked cache alone on approxH scored 41,350.5, and 41,360.7 on a second node, +81.9% over approxH measured in the same job. Most of `nrc`'s gain over approxN is the cache.
- On a collaborator's H100s `nrc` measured 45,594, 6% lower, with identical images.
- "FLIP" rows passed the FLIP rule of their time. Under the final gate only approxH, approxI, approxN and `nrc` were judged; all four pass.

### Run settings

Settings that are not compile flags:

- `configs/nrc_env.txt`: every scene. 16x concurrent states, cheaper queue scans, 2x shadow pool, the frozen cache on rule c0.
- `configs/<scene>/nrc_env.txt`: per scene. BVH2 build options, BVH4 on junkshop, flat triangles, sort partitions, shadow compaction, 24x states on classroom.
- `configs/scene_settings.py`: Blender settings applied before the benchmark script. Passes off, light threshold, scrambling distance, glossy bounces.
- `configs/exact_env.txt` and `configs/<scene>/exact_env.txt`: the exact configuration. 8x states, BVH2 leaf size 2, spatial splits and 131,072-path sort partitions on monster and classroom; scene settings untouched. Its per-scene scores were about 6,706, 4,050 and 3,356.

The compile-time KernelData in `kdata_const.h` was dumped with exactly these scene settings. A cubin built with it ignores later changes to the folded fields: scrambling, light threshold, bounces, clamps, filter glossy, light tree and the seed. 22 settings sweeps on such cubins measured nothing before this was found.

## 5. The baked cache

The cache maps a surface point and direction to outgoing radiance. Under rule c0 every camera path queries it at its first hit and ends there.

| | |
|---|---|
| encoding | position: 16-level hash grid, 2^24 entries per level, resolution 128 to 2048; five more inputs one-blob encoded; the rest passed through |
| network | fully fused MLP, 64 wide, 5 hidden layers, ReLU |
| training data | 9 exact renders at 2048 spp per scene; one record per path vertex: features and the one-sample estimate of incident radiance from the full path suffix |
| optimizer | Adam with EMA, five stages: lr 3e-3 to 1e-4, beta2 0.999 then 0.9999, L2 loss; every record used once |
| calibration | one global output scale per rule, fitted so the mean luminance of a seed-3 render matches an exact seed-2 render; the c0 scales are 1.11, 1.08 and 1.24 |
| inference | frozen weights; fused wgmma kernel; 11.5% of monster's kernel time on approxH with the cache |
| cost | 9 to 18 minutes per scene on one H100; 0.9 GB per scene |

The bakes are public: Hugging Face `wchai/cycles-h100-nrc-bake`, `nrc_bakehi/<scene>_t24/bake_c0.nrc`. `configs/bakes.sha256` holds their hashes.

The bake has its own build: the exact configuration's kernel flags plus `-DCYCLES_NRC` on the tree of `patches/bake/kernel.diff`, built with `FLAGS=exact_nvcc_flags.txt EXTRA=-DCYCLES_NRC scripts/build_cubins.sh`, and a host with the training hooks, `patches/bake/host.diff`. Built from this repository, the bake cubins are byte-identical to the project's (`configs/bake_cubins.sha256`), and the public bakes match the files the 48,641 run loaded (`configs/bakes.sha256`). The bake scripts are in `nrc/bake/`; no bake has been re-run from them.

Why the cache passes now and failed before: its error is a smooth bias, the monster body 10-15% low and the walls 5-10% high, not noise. More capacity does not remove it: a 32x larger hash table and a 128-wide MLP left the bias in place, and the 128-wide network cost 15-35% more time on the unfused path. On held-out cache error ReLU beat SiLU at every width. The FLIP rule caught the bias; the 32-spp PSNR gate does not, because an exact 32-spp render is far noisier than the bias.

## 6. Reproduce

```bash
export BENCH=$W/bench/blender-benchmark-launcher BLENDER=$W/host/install
export CUBINS=$W/nrc_cubins ENGINE=$W/engine/libcycles_nrc.so BAKES=$W/bakes
CUDA_VISIBLE_DEVICES=0 scripts/run.sh nrc bench out/nrc_1        # repeat for n = 6, one GPU each
CUDA_VISIBLE_DEVICES=0 scripts/run.sh nrc check out/nrc          # 4096-spp gate images
scripts/make_reference.sh $W/blender-5.3.0-linux-x64 ref 0    # reference + bar, ~2 GPU-h
python3 scripts/gate.py ref out/nrc
```

The `exact` configuration needs no engine, bake or reference. Its check is the image difference against the official binary, measured against the seed-to-seed noise floor:

```bash
export BENCH=$W/bench/blender-benchmark-launcher BLENDER=$W/host/install CUBINS=$W/exact_cubins
CUDA_VISIBLE_DEVICES=0 scripts/run.sh exact bench out/exact_1    # repeat for n = 6
CUDA_VISIBLE_DEVICES=0 scripts/run.sh exact check out/exact      # 32 spp, seed 0
export STOCK=$W/blender-5.3.0-linux-x64                          # the official binary
BLENDER=$STOCK CUDA_VISIBLE_DEVICES=0 scripts/run.sh stock check out/stock0
BLENDER=$STOCK CUDA_VISIBLE_DEVICES=0 SEED=1 scripts/run.sh stock check out/stock1
python3 scripts/identity.py out/stock0 out/stock1 out/exact     # project: ratios 1.4e-3 to 2.8e-3
```

On Slurm, `sbatch slurm/repro.sbatch <repo> $W [nrc|exact]` runs six measurements and the check images of one configuration on one 8-GPU node and writes them to `$W/runs/<job id>`.

Reproduction from this repository, with the host, the cubins and the engine built by the scripts above and the public bakes:

| | monster | junkshop | classroom | score |
|---|---|---|---|---|
| nrc, project run | 20,274.8 | 13,319.5 | 15,047.0 | 48,641.3 |
| nrc, this repository | 20,296.7 | 13,323.4 | 15,100.9 | 48,721.0 |
| exact, project run | 6,706 | 4,050 | 3,356 | 14,106 |
| exact, this repository | 6,711.2 | 4,049.0 | 3,355.6 | 14,115.8 |

Each row is the median of six official measurements on one 8-GPU H100 node; the differences are +0.16% and +0.07%, inside the node spread of about 0.3%. For nrc, the gate returns the same PSNR and luminance as section 2 on all three scenes; it used the project's reference images, since `make_reference.sh` has not been run. For exact, the image differs from the official binary's by 1.4e-3, 2.5e-3 and 2.8e-3 of the seed-to-seed noise floor, the project's values. The nine cubins, nrc, exact and bake, rebuild byte-identical. Those two runs used the scripts before the configuration names `nrc` and `exact` and the single work-directory layout; the renamed scripts were checked by a dry run of the environment they set, not on a GPU.

### Samples per minute and image quality

The score counts samples, and an `nrc` sample ends at its first hit. To see what that buys, we rendered every scene at seed 0 with `nrc` at 16 to 1,024 samples and with `exact` at 32 to 512 samples (`CHECK_SPP=<n> scripts/run.sh <cfg> check`), and measured PSNR against the 65,536-spp reference and Cycles' render time.

| scene | `nrc` samples | PSNR | luminance | render time | `exact` to the same PSNR | stock Blender to the same PSNR |
|---|---|---|---|---|---|---|
| monster | 64 | 28.5 dB | -1.29% | 0.20 s | 0.86 s | 2.1 s |
| monster | 256 | 31.5 dB | -0.62% | 0.68 s | 1.56 s | 3.8 s |
| monster | 1,024 | 32.8 dB | +0.48% | 2.79 s | 2.04 s | 4.9 s |
| junkshop | 64 | 21.8 dB | -2.17% | 0.33 s | 1.02 s | 1.7 s |
| junkshop | 256 | 23.4 dB | -0.30% | 1.13 s | 1.27 s | 2.1 s |
| junkshop | 1,024 | 23.7 dB | +0.07% | 4.40 s | 1.33 s | 2.2 s |
| classroom | 64 | 28.2 dB | +1.49% | 0.29 s | 1.85 s | 3.4 s |
| classroom | 256 | 31.2 dB | +0.58% | 1.03 s | 3.33 s | 6.1 s |
| classroom | 1,024 | 32.1 dB | +0.13% | 4.06 s | 3.99 s | 7.3 s |

The `exact` times interpolate its measured points in log time; PSNR rises 3.0 to 3.4 dB per doubling of samples there. Stock Blender renders the same image as `exact`, and its times are `exact`'s scaled by their samples-per-minute ratio, not measured. The cache's quality levels off near 33, 24 and 32 dB, the values of the 4,096-sample gate images, while `exact` reaches 45, 42 and 45 dB at 4,096 samples. So `nrc` gives a usable image sooner than `exact` and stock below that level, and is slower above it; at the gate's own 4,096 samples an `exact` render of the same PSNR takes 1.3 to 4.0 s against 12 to 19 s. Under 1,024 samples its mean luminance also strays past 0.5% on at least one scene.

Against the PSNR bar alone, a second run found the fewest samples at which each configuration clears it, three seed-0 renders per point, median render time. The exact build and stock Blender sit on the bar at 32 samples, since the bar is the exact 32-sample PSNR.

| scene | `nrc` | `exact`, 32 samples | stock Blender, 32 samples | `nrc` sooner than `exact` / stock |
|---|---|---|---|---|
| monster | 20 samples, 24.51 dB, 0.074 s | 0.373 s | 0.789 s | 5.0x / 10.7x |
| junkshop | 20 samples, 18.85 dB, 0.133 s | 0.636 s | 0.988 s | 4.8x / 7.4x |
| classroom | 24 samples, 23.99 dB, 0.132 s | 0.938 s | 1.499 s | 7.1x / 11.4x |

Summed over the three scenes this is 28.6 images per second for `nrc`, 5.32 for `exact` and 2.95 for stock Blender, 9.7x and 1.8x. At these sample counts `nrc`'s mean luminance is off by +2.0%, -6.1% and +8.2%, so under the full gate, with the 0.5% luminance limit, a passing image needs about 1,024 samples, where `nrc` is no faster than `exact`.

## 7. What did not work

| idea | result | why |
|---|---|---|
| wide BVH4 for closest rays | predicted 1.22-1.24x traversal, below the 1.25x go bar; BVH4 kept only for junkshop shadow rays | a tuned 4-wide node costs 121 SASS instructions against 66 for BVH2, so the instruction count stays flat |
| main and shadow kernels on two streams | -0.6% to -0.8% | the idle time it could fill was estimated at 1.6-2.2% |
| closures in registers, per-material kernels | predicted +1.8% to +4.8%, not built | 36-58 more registers on kernels already at 64; per-material programs with closures in local memory gave +4.3% instead |
| material and ray-octant sort | slower on H100 | it cut warp steps by 4-12% on A40; the H100 measurement lost |
| batched texture fetches | -0.43% | |
| SSS empty-space skip | loses | the traversal sits inside the skip loop, so a warp runs it once per distinct iteration |
| single-closure NEE | no gain | MIS still needs every closure's pdf |
| path drains, host gaps, bookkeeping | 94-96% of the timed render is already integrator kernels | no scheduling-only route |
| online neural cache, trained during the render | fails FLIP unless it ends very few paths | smooth local bias of 1.4-2.6% RMS at a 9-px scale |
| wider MLP for the baked cache | no image gain, 15-35% slower on the unfused path | the bias is not a capacity problem |
| smaller bakes | t22 -0.56% | t24 stays |
| Russian roulette exponent 2 | fails the image rule on monster and classroom | |
| Sobol cap tuned on seed pair 0/1 | +1.5%, fails 8 of 9 other seed pairs | the FLIP drop at seeds 0/1 was chance |
| per-cell mean cache ending at the first vertex on monster | +28% on monster, luminance -2.5% | |
| tuning results from an H200 | did not transfer | same die, but 3.35 TB/s HBM on H100 against 4.8 TB/s makes spills cost more |

## 8. Limits

- Seeds: at seeds 2, 4 and 6 the baked cache on approxH gives monster luminance -0.65%, +0.38% and +0.24%, so it fails at seed 2. The `nrc` monster and junkshop cubins fold the seed to 0, so `nrc` was checked at seed 0 only.
- Sobol caps: the `nrc` cubins keep the Sobol table caps of approxH and approxI, 1024 on monster and 256 on classroom. Their FLIP benefit at seeds 0 and 1 was chance; `nrc` was not measured without them.
- The gate: the final number depends on the 32-spp gate. Under the FLIP rule used until the last evening, the best official scores were about 23,000; approxN and the fast cache rules were judged only under the later rules.
- No ablation of `nrc`: which earlier steps still matter under rule c0 is not measured. Under c0 the deep-bounce gains of roulette, SSS walk and bounce limits should largely vanish.
- Per-scene choices can come from scene content instead of the scene's name: on the exact track a rule-based configuration matched the hand-picked one within 0.06%; on the approximate track it was 0.9% slower.
- Open: kernel variants that drop work rule c0 makes dead, per-scene scrambling 0 or a higher light threshold on the cache arm with new KernelData headers (+3% to +7% per scene in screens), and the smaller emitter-bake gain on the collaborator's classroom.
