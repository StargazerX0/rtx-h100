# Smoke test of the Cycles-facing engine .so (dlopen, as Cycles does): primary context, non-blocking stream,
# nrc_create(config) -> nrc_train -> nrc_infer -> nrc_destroy. Usage: python3 so_smoke.py <lib.so> <config.json>
import ctypes as C, random, struct, sys
cu = C.CDLL("libcuda.so.1")
def ck(r, what):
    if r: raise SystemExit(f"{what}: CUDA error {r}")
ck(cu.cuInit(0), "cuInit"); dev = C.c_int(); ck(cu.cuDeviceGet(C.byref(dev), 0), "cuDeviceGet")
ctx = C.c_void_p(); ck(cu.cuDevicePrimaryCtxRetain(C.byref(ctx), dev), "retain"); ck(cu.cuCtxPushCurrent(ctx), "push")
st = C.c_void_p(); ck(cu.cuStreamCreate(C.byref(st), 1), "stream")
lib = C.CDLL(sys.argv[1]); lib.nrc_create.restype = C.c_void_p
lib.nrc_infer.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p]
lib.nrc_train.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_void_p]
lib.nrc_destroy.argtypes = [C.c_void_p]
e = lib.nrc_create(open(sys.argv[2]).read().encode(), st)
if not e: raise SystemExit("nrc_create returned NULL")
n = 65536; rnd = random.Random(1)
feats = b"".join(struct.pack("<16e", *([rnd.random() for _ in range(14)] + [0, 0])) for _ in range(n))
tgt = struct.pack(f"<{3*n}f", *[rnd.random() for _ in range(3 * n)])
din, dt, dout = C.c_uint64(), C.c_uint64(), C.c_uint64()
for p, b in ((din, len(feats)), (dt, len(tgt)), (dout, 12 * n)): ck(cu.cuMemAlloc_v2(C.byref(p), C.c_size_t(b)), "alloc")
ck(cu.cuMemcpyHtoD_v2(din, feats, C.c_size_t(len(feats))), "h2d"); ck(cu.cuMemcpyHtoD_v2(dt, tgt, C.c_size_t(len(tgt))), "h2d")
for _ in range(50): assert lib.nrc_train(e, n, din.value, dt.value, 4, st) == 0
assert lib.nrc_infer(e, n, din.value, dout.value, st) == 0
ck(cu.cuStreamSynchronize(st), "sync")
out = (C.c_float * (3 * n))(); ck(cu.cuMemcpyDtoH_v2(out, dout, C.c_size_t(12 * n)), "d2h")
vals = list(out); mean = sum(vals) / len(vals)
ok = all(v == v and 0 <= v < 1e3 for v in vals) and 0.2 < mean < 0.8
print(f"SO_SMOKE {'PASS' if ok else 'FAIL'}: {n} queries after 50 train calls, mean prediction {mean:.4f} (targets U(0,1): 0.5)")
lib.nrc_destroy(e)
