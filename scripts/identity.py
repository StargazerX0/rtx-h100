"""Exactness check: is a configuration's image equal to stock up to floating-point rounding?
  identity.py <stock seed-0 dir> <stock seed-1 dir> <variant dir>      (each holds <scene>_s32.exr; run.sh ... check)
Per scene: ratio = MAE(variant, stock seed 0) / MAE(stock seed 1, stock seed 0), the difference against the
seed-to-seed noise floor, and tail = the share of pixels whose difference exceeds the local seed-to-seed difference
(9x9 mean). FP-level changes give a ratio of 3e-5 to 3e-3 and a tail under 1e-3; a wrong render gives a ratio near 1.
Exit 1 if a scene's ratio is above 1e-2, or if the noise floor is zero (the two stock seeds rendered the same image)."""
import os, sys
import numpy as np, OpenEXR

SC = ("monster", "junkshop", "classroom")


def load(p):
    ch = OpenEXR.File(p).parts[0].channels
    x = ch["RGB"].pixels if "RGB" in ch else ch["RGBA"].pixels[..., :3] if "RGBA" in ch else np.dstack([ch[c].pixels for c in "RGB"])
    return np.nan_to_num(x.astype(np.float64), nan=0, posinf=0, neginf=0)


def local_mean(img, k=9):
    """Box mean over k x k pixels (integral image)."""
    pad = k // 2
    q = np.pad(img, pad, mode="edge")
    c = np.zeros((q.shape[0] + 1, q.shape[1] + 1)); c[1:, 1:] = q.cumsum(0).cumsum(1)
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


ref, noise, var = sys.argv[1:4]
ok = True
print(f"{'scene':10s} {'floor MAE':>10s} {'MAE':>10s} {'ratio':>9s} {'tail':>9s}  result")
for s in SC:
    f = f"{s}_s32.exr"
    if not all(os.path.exists(os.path.join(d, f)) for d in (ref, noise, var)):
        print(f"{s:10s} missing"); ok = False; continue
    r, n, x = (load(os.path.join(d, f)) for d in (ref, noise, var))
    floor = np.abs(n - r).mean(); d = np.abs(x - r); mae = d.mean()
    tail = float((d.max(axis=2) > local_mean(np.abs(n - r).max(axis=2)) + 1e-6).mean())
    good = floor > 0 and mae / floor <= 1e-2
    ok &= good
    print(f"{s:10s} {floor:10.3e} {mae:10.3e} {mae / max(floor, 1e-30):9.2e} {tail:9.2e}  {'exact' if good else 'FAIL'}")
sys.exit(0 if ok else 1)
