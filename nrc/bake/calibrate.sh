#!/bin/bash
# Calibrate a bake for rule c0 (every path ends at its first hit) on one GPU: calibrate.sh <scene> <bake dir>
# -> <bake dir>/bake_c0.nrc with output_scale = (E - Z) / (A - Z) (set_scale.py). E: exact render, seed 2.
# A: the bake at scale 1, seed 3. Z: a zero cache, seed 3. 1024 spp each. Seeds 2-3 never enter the gate.
# The project rendered E with the exact cubin without the cache code; here E uses the bake cubin with the cache off.
# Env: as render.sh, plus ENGINE; RUNCFG (default run_wgmma.json for H100; run_mma.json on sm_86).
# The project's c0 scales were 1.11, 1.08 and 1.24 (monster, junkshop, classroom).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); SC=${1:?scene}; S=$(cd "${2:?bake dir}" && pwd); export OUT=$S/cal; mkdir -p "$OUT"
: "${ENGINE:?}"
COM=(CYCLES_NRC=1 CYCLES_NRC_STATS=1 CYCLES_NRC_FREEZE=1 CYCLES_NRC_C=0)
bash "$HERE/render.sh" "$SC" exact_s2 2 1024 || exit 1
bash "$HERE/render.sh" "$SC" bake_s3 3 1024 "${COM[@]}" CYCLES_NRC_ENGINE=$ENGINE \
  CYCLES_NRC_CONFIG=${RUNCFG:-$HERE/../configs/run_wgmma.json} CYCLES_NRC_LOAD=$S/bake.nrc || exit 1
bash "$HERE/render.sh" "$SC" zero_s3 3 1024 "${COM[@]}" CYCLES_NRC_ENGINE=zero || exit 1
python3 "$HERE/set_scale.py" "$S/bake.nrc" "$S/bake_c0.nrc" "$OUT/$SC-exact_s20001.exr" "$OUT/$SC-bake_s30001.exr" \
  "$OUT/$SC-zero_s30001.exr" c0 | tee -a "$S/calibration.tsv"
