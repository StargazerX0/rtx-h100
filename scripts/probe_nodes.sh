#!/bin/bash
# probe_nodes.sh <blender install> <scene> <out dir>     (BENCH = the launcher data dir; CPU only, ~1 min)
# Renders one sample at 10% resolution on the CPU with the Cycles info log. The patched host prints the kernel feature
# mask, the SVM node types the scene uses and its maximum closure count; scripts/svm_nodes_header.py turns the log into
# configs/<scene>/svm_nodes_used.h, and the mask and count go into the scene's nvcc flags.
set -euo pipefail
B=${1:?install}; SC=${2:?scene}; OUT=${3:?out}; : "${BENCH:?}"; mkdir -p "$OUT"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
blend=$(find "$BENCH/scenes" -path "*/$SC/main.blend" | head -1)
XDG_CACHE_HOME=$T/xdg CUDA_VISIBLE_DEVICES= "$B/blender" -b "$blend" --log-level info -t 8 --python-expr "
import bpy
sc = bpy.context.scene
sc.cycles.device = 'CPU'; sc.cycles.samples = 1; sc.render.resolution_percentage = 10
sc.render.filepath = '$T/probe_'
" -f 1 > "$OUT/probe_$SC.log" 2>&1 || true
grep -h "Kernel features mask\|SVM nodes used\|Max closures" "$OUT/probe_$SC.log"
