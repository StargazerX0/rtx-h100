#!/bin/bash
# One render for the bake pipeline, on one GPU (CUDA_VISIBLE_DEVICES), with the bake host and the scene's bake cubin.
#   render.sh <scene> <tag> <seed> <spp> [ENV=VAL ...]       -> $OUT/<scene>-<tag>0001.exr and .log
# Env: BAKE_BLENDER (install dir, patches/bake/host.diff), BAKE_CUBINS (dir with <scene>.cubin built from
# patches/bake/kernel.diff: FLAGS=exact_nvcc_flags.txt EXTRA=-DCYCLES_NRC build_cubins.sh), BENCH (launcher data dir), OUT,
# JITTER=<k> (camera shift by hash k, up to +-0.002 of the frame). Without CYCLES_NRC=1 the render is exact.
set -uo pipefail
: "${BAKE_BLENDER:?}" "${BAKE_CUBINS:?}" "${BENCH:?}" "${OUT:?}" "${CUDA_VISIBLE_DEVICES:?set it to one GPU index}"; mkdir -p "$OUT"; JIT=${JITTER:-0}
export TMPDIR=${TMPDIR:-$OUT/tmp}/render-$$; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
SC=$1 tag=$2 seed=$3 spp=$4; shift 4
blend=$(find "$BENCH/scenes" -path "*/$SC/main.blend" | head -1)
case $SC in
  monster|classroom) HOST=(CYCLES_CONCURRENT_STATES_FACTOR=8 CYCLES_BVH2_SPATIAL_SPLIT=1 CYCLES_BVH2_MAX_TRIANGLE_LEAF_SIZE=2 CYCLES_SORT_PARTITION_SIZE=131072) ;;
  junkshop) HOST=(CYCLES_CONCURRENT_STATES_FACTOR=8 CYCLES_BVH2_MAX_TRIANGLE_LEAF_SIZE=2) ;;
esac
env XDG_CACHE_HOME=$TMPDIR/xdg CYCLES_CUDA_CUBIN=$BAKE_CUBINS/$SC.cubin CYCLES_TIMER_REPORT=1 "${HOST[@]}" "$@" \
  "$BAKE_BLENDER/blender" -b "$blend" -t 4 --python-expr "
import bpy, random
sc = bpy.context.scene
ad = sc.animation_data   # classroom keys cycles.seed with an fcurve that would override the seed
if ad and ad.action:
    act = ad.action
    bags = [act] if hasattr(act, 'fcurves') else [cb for l in act.layers for st in l.strips for cb in getattr(st, 'channelbags', [])]
    for bag in bags:
        fc = bag.fcurves.find('cycles.seed')
        if fc: bag.fcurves.remove(fc)
sc.cycles.use_animated_seed = False; sc.render.use_compositing = False; sc.render.use_sequencer = False
pr = bpy.context.preferences.addons['cycles'].preferences
pr.compute_device_type = 'CUDA'; pr.get_devices()
for d in pr.devices: d.use = d.type == 'CUDA'
sc.cycles.device = 'GPU'; sc.cycles.samples = $spp; sc.cycles.seed = $seed; sc.render.resolution_percentage = 100
sc.cycles.use_denoising = False; sc.cycles.use_adaptive_sampling = False; sc.cycles.use_auto_tile = False
if $JIT:
    r = random.Random($JIT); cam = sc.camera.data
    cam.shift_x += r.uniform(-0.002, 0.002); cam.shift_y += r.uniform(-0.002, 0.002)
sc.render.image_settings.file_format = 'OPEN_EXR'; sc.render.image_settings.color_depth = '32'
sc.render.filepath = '$OUT/$SC-$tag'
" -f 1 > "$OUT/$SC-$tag.log" 2>&1
rc=$?
echo "$SC-$tag rc=$rc $(grep -h 'cycles_nrc: \(loaded\|saved\)' "$OUT/$SC-$tag.log" | tr '\n' ' ')"
exit $rc
