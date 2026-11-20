#!/bin/bash
# Bake one scene's radiance cache on one GPU (9-18 min on an H100): bake.sh <scene> <out dir>  -> <out>/bake.nrc
# Env: as render.sh, plus ENGINE (libcycles_nrc.so from scripts/build_engine.sh); MODEL (default nrc/configs/t24.json).
# Each stage renders the scene exactly at 2048 spp (CYCLES_NRC_TERM=0: no path ends in the cache) and feeds the engine
# fresh training records: per path vertex, the features and the one-sample estimate of incident radiance from the full
# path suffix, with Cycles' own bounce limits and Russian roulette, so the target is unbiased. Every record is used once.
# A stage starts from the previous stage's save with a fresh optimizer:
#   1: seed 61, Adam lr 3e-3, beta2 0.999, EMA 0.99, 8 steps per call
#   2-5: two renders each, lr 3e-3, 1e-3, 3e-4, 1e-4; beta2 0.9999; EMA 0.995, 0.999, 0.9995, 0.9999; 16 steps per call
# Seeds >= 63 also shift the camera by up to +-0.002 of the frame.
# Loss is plain L2: with relative L2 early bakes drifted 1.7-3.2x high on monster and junkshop.
# lr 1e-2 sometimes killed the network on junkshop and classroom: every ReLU dead.
# Not ported from the project's script: the held-out error check on dumped records after each stage.
# One record per vertex (CYCLES_NRC_TRAIN_WEIGHT=0): throughput resampling made deep vertices rare and the cache ~20% low.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); SC=${1:?scene}; S=${2:?out dir}; mkdir -p "$S"; S=$(cd "$S" && pwd)
: "${ENGINE:?}"; export OUT=$S/renders; mkdir -p "$OUT"
MODEL=${MODEL:-$HERE/../configs/t24.json}
BASE=(CYCLES_NRC=1 CYCLES_NRC_STATS=1 CYCLES_NRC_ENGINE=$ENGINE CYCLES_NRC_TERM=0 CYCLES_NRC_TRAIN_FRAC=0.25
      CYCLES_NRC_SLOTS=8388608 CYCLES_NRC_RECORDS=16777216 CYCLES_NRC_TRAIN_SLOTS=1048576 CYCLES_NRC_TRAIN_WEIGHT=0)
opt() {  # opt <lr> <beta2> <ema> -> config file with that optimizer and L2 loss
  local f=$S/opt_$1_$2_$3.json
  python3 -c "
import json
m = json.load(open('$MODEL'))
m['optimizer'] = {'otype': 'EMA', 'decay': $3, 'nested': {'otype': 'Adam', 'learning_rate': $1, 'beta1': 0.9, 'beta2': $2, 'epsilon': 1e-15, 'l2_reg': 1e-6}}
m['loss'] = {'otype': 'L2'}
m['nrc'] = {'warmup_infer_batch': 262144, 'warmup_train_batch': 262144}
json.dump(m, open('$f', 'w'))" || exit 1
  echo "$f"
}
prev=""; t0=$(date +%s)
for st in 1 2 3 4 5; do
  case $st in
    1) seeds=(61); cfg=$(opt 3e-3 0.999 0.99); STEPS=8 ;;
    2) seeds=(62 63); cfg=$(opt 3e-3 0.9999 0.995); STEPS=16 ;;
    3) seeds=(64 65); cfg=$(opt 1e-3 0.9999 0.999); STEPS=16 ;;
    4) seeds=(66 67); cfg=$(opt 3e-4 0.9999 0.9995); STEPS=16 ;;
    5) seeds=(68 69); cfg=$(opt 1e-4 0.9999 0.9999); STEPS=16 ;;
  esac
  [ -s "$cfg" ] || { echo "no optimizer config"; exit 1; }
  for sd in "${seeds[@]}"; do
    j=0; [ "$sd" -ge 63 ] && j=$sd
    out=$S/st$st-s$sd.nrc; L=(); [ -n "$prev" ] && L=(CYCLES_NRC_LOAD=$prev)
    JITTER=$j bash "$HERE/render.sh" "$SC" "bake_st$st-s$sd" "$sd" 2048 "${BASE[@]}" CYCLES_NRC_TRAIN_STEPS=$STEPS "${L[@]}" \
      CYCLES_NRC_CONFIG=$cfg CYCLES_NRC_SAVE=$out \
      CYCLES_NRC_META="{\"scene\": \"$SC\", \"stage\": $st, \"seed\": $sd, \"spp\": 2048, \"frac\": 0.25, \"from\": \"$(basename "${prev:-scratch}")\"}" || exit 1
    rm -f "$OUT/$SC-bake_st$st-s${sd}0001.exr"
    [ -s "$out" ] || { echo "no save $out"; exit 1; }
    prev=$out
  done
done
cp "$prev" "$S/bake.nrc"; cp "$MODEL" "$S/config.json"
echo "bake $(( $(date +%s) - t0 )) s: $S/bake.nrc"
