# Warm repeats (ours, for screening only): run the official benchmark script BENCH_REPEATS times in one Blender
# process after one .blend load; each run has its own 30 s timed window and prints its own "Rendered N samples"
# line. BENCH_PERSIST=1 also turns on Cycles persistent data so later renders reuse the session and BVH.
import os, runpy, sys
import bpy
n = int(os.environ.get("BENCH_REPEATS", "1"))
main = os.environ["BENCH_MAIN"]
if os.environ.get("BENCH_PERSIST") == "1":
    for sc in bpy.data.scenes:
        sc.render.use_persistent_data = True
sys.path.insert(0, os.path.dirname(main))
for i in range(n):
    print(f"BENCH_REPEAT {i}", flush=True)
    runpy.run_path(main, run_name="__main__")
