"""Per-rule calibration of a bake: output_scale s = (E - Z) / (A - Z) from mean image luminance, where E is an exact
render at seed 2, and A (the bake at scale 1) and Z (a zero cache) are renders at seed 3 under the same rule.
The cached part of the image is A - Z; the exact image needs E - Z. Writes <out.nrc> = <in.nrc> with header fields
"output_scale" and "calibration"; the weights are unchanged.
  set_scale.py <in.nrc> <out.nrc> <E.exr> <A.exr> <Z.exr> <rule>"""
import json, struct, sys
import numpy as np, OpenEXR


def lum(path):
    ch = OpenEXR.File(path).parts[0].channels
    x = ch["RGB"].pixels if "RGB" in ch else ch["RGBA"].pixels[..., :3] if "RGBA" in ch else np.dstack([ch[c].pixels for c in "RGB"])
    x = np.nan_to_num(x.astype(np.float32), nan=0, posinf=0, neginf=0)
    return float(np.mean(0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]))


src, dst, E, A, Z, rule = sys.argv[1:7]
e, a, z = lum(E), lum(A), lum(Z)
s = (e - z) / (a - z)
raw = open(src, "rb").read()
assert raw[:8] == b"NRCBAKE1"
hl = struct.unpack("<Q", raw[8:16])[0]
h = json.loads(raw[16:16 + hl])
h["output_scale"] = s
h["calibration"] = {"rule": rule, "E": e, "A": a, "Z": z, "cached_share": (a - z) / e}
hs = json.dumps(h).encode()
open(dst, "wb").write(raw[:8] + struct.pack("<Q", len(hs)) + hs + raw[16 + hl:])
print(f"{rule}\tscale {s:.4f}\tcached share {(a - z) / e:.3f}\tA/E {a / e:.4f}")
