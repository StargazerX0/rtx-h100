"""Acceptance gate for an approximate configuration, per scene:
  1. PSNR(approx 4096 spp, seed 0  vs  65536-spp exact reference) >= mean PSNR(exact 32 spp vs the same reference);
  2. |mean luminance of the approx image / mean luminance of the reference - 1| <= 0.5%.
Reference = mean of 4 exact 16384-spp renders (seeds 100-103); the 32-spp bar = mean over 8 exact renders (seeds 200-207).
PSNR on display-mapped images: y = x*k / (1 + x*k), k = 0.5 / median luminance of the reference, then gamma 1/2.2.
  gate.py <ref dir> <check dir> [workers]
<ref dir>: r64k_<k>/<scene>_s16384.exr and e32_<k>/<scene>_s32.exr (scripts/make_reference.sh);
<check dir>: <scene>_s4096.exr (run.sh nrc check). Exit 1 if any scene fails."""
import glob, os, sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np, OpenEXR

SC = ("monster", "junkshop", "classroom")
W = np.array([0.2126, 0.7152, 0.0722], np.float32)


def load(p):
    ch = OpenEXR.File(p).parts[0].channels
    x = ch["RGB"].pixels if "RGB" in ch else ch["RGBA"].pixels[..., :3] if "RGBA" in ch else np.dstack([ch[c].pixels for c in "RGB"])
    return np.nan_to_num(x.astype(np.float32), nan=0, posinf=0, neginf=0)


def disp(x, k):
    y = x * k
    y = y / (1 + y)
    return np.clip(y, 0, 1) ** (1 / 2.2)


def psnr(a, b):
    return 10 * np.log10(1.0 / max(float(np.mean((a - b) ** 2)), 1e-12))


def one(args):
    ref_dir, check_dir, s = args
    refs = sorted(glob.glob(f"{ref_dir}/r64k_*/{s}_s16384.exr"))
    test = f"{check_dir}/{s}_s4096.exr"
    bars = sorted(glob.glob(f"{ref_dir}/e32_*/{s}_s32.exr"))
    if len(refs) != 4 or len(bars) != 8 or not os.path.exists(test):   # an incomplete reference would move the bar
        return s, None
    ref = sum(load(p) for p in refs) / len(refs)
    k = 0.5 / max(float(np.median(ref @ W)), 1e-4)
    dr = disp(ref, k)
    bar = float(np.mean([psnr(dr, disp(load(p), k)) for p in bars]))
    t = load(test)
    return s, (psnr(dr, disp(t, k)), bar, float(np.mean(t @ W)) / float(np.mean(ref @ W)) - 1)


def main():
    ref_dir, check_dir = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    with ProcessPoolExecutor(n, mp_context=mp.get_context("spawn")) as ex:
        res = dict(ex.map(one, [(ref_dir, check_dir, s) for s in SC]))
    ok = True
    print(f"{'scene':10s} {'PSNR dB':>8s} {'bar dB':>8s} {'luminance':>10s}  result")
    for s in SC:
        if res[s] is None:
            print(f"{s:10s} missing: needs 4 reference renders, 8 bar renders and the check image"); ok = False; continue
        p, bar, dl = res[s]
        good = p >= bar and abs(dl) <= 0.005
        ok &= good
        print(f"{s:10s} {p:8.2f} {bar:8.2f} {100 * dl:+9.2f}%  {'pass' if good else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
