// nrc_test: standalone harness for the NRC engine (include/cycles_nrc.h).
//   1. Sets up CUDA like Cycles 5.3.0: driver API, primary context pushed, CU_STREAM_NON_BLOCKING stream.
//   2. Trains on a synthetic radiance field with noisy (Monte Carlo-like) targets; reports held-out relative MSE
//      against the clean field. Pass: final relMSE < 0.25 x initial.
//   3. Compares nrc_infer with a CPU reference (hash grid + one-blob + identity + MLP from the EMA weights).
//   4. With --engine wgmma|mma: compares the fused engine with tcnn's inference on the same EMA weights (XCHECK).
//   5. Measures nrc_infer queries/s and nrc_train records/s for batches 2^16..2^21 (and tcnn's inference on the
//      same weights when the engine is fused).
// Usage: nrc_test [--hidden H] [--act ReLU|SiLU] [--log2t T] [--iters N] [--bench-only] [--no-bench]
//                 [--tag NAME] [--device D] [--engine tcnn|wgmma|mma] [--fused-wg 0|2|4]
//                 [--grid-levels L]  (fused engine: 0 = gathers in the kernel, L = split grid pass of L levels, -1 = by L2 size)
#include "cycles_nrc.h"
#include "../src/cycles_nrc_debug.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CU(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s_; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)
#define CK(x) do { cudaError_t r_ = (x); if (r_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, \
	cudaGetErrorString(r_)); exit(1); } } while (0)

static const float PI = 3.14159265358979f;

struct Rec { float f[14]; float L[3]; };

// Synthetic radiance field: diffuse part with position detail (incl. a hard edge), specular lobe around the normal.
static void field(const float* f, float* L) {
	const float px = f[0], py = f[1], pz = f[2];
	const float E = 1.5f + 0.6f * sinf(2 * PI * 3 * px) * sinf(2 * PI * 2 * py) + 0.4f * cosf(2 * PI * 5 * pz) +
	                0.5f * (px > 0.5f ? 1.f : 0.f);
	const float dt = f[3] - f[5], dp = f[4] - f[6];
	const float S = (1 - f[7]) * 4.0f * expf(-20.0f * (dt * dt + dp * dp)) + 0.2f;
	for (int c = 0; c < 3; ++c) L[c] = f[8 + c] * E + f[11 + c] * S;
}

static void make_records(size_t n, uint64_t seed, std::vector<__half>& in, std::vector<float>& clean, std::vector<float>& noisy) {
	std::mt19937_64 rng(seed);
	std::uniform_real_distribution<float> U(0.f, 1.f);
	in.assign(n * NRC_IN, __float2half(0.f));
	clean.resize(n * 3);
	noisy.resize(n * 3);
	for (size_t i = 0; i < n; ++i) {
		float f[14];
		for (int d = 0; d < 14; ++d) f[d] = U(rng);
		for (int d = 8; d < 14; ++d) f[d] *= 0.9f;
		for (int d = 0; d < 14; ++d) { in[i * NRC_IN + d] = __float2half(f[d]); f[d] = __half2float(in[i * NRC_IN + d]); }
		float L[3];
		field(f, L);
		const float g = 0.5f + U(rng);  // one-sample estimate: mean 1, relative variance 1/12
		for (int c = 0; c < 3; ++c) { clean[i * 3 + c] = L[c]; noisy[i * 3 + c] = L[c] * g; }
	}
}

static double rel_mse(const std::vector<float>& pred, const std::vector<float>& ref, size_t n) {
	double s = 0;
	for (size_t i = 0; i < n * 3; ++i) { const double d = pred[i] - ref[i]; s += d * d / (ref[i] * ref[i] + 0.01); }
	return s / (n * 3);
}

// ---------------- CPU reference of the default encoding + FullyFusedMLP ----------------
static inline float h(float x) { return __half2float(__float2half(x)); }
static inline float quartic_cdf(float x, float inv_radius) {
	const float u = x * inv_radius, u2 = u * u, u4 = u2 * u2;
	return std::max(0.0f, std::min(1.0f, (15.f / 16.f) * u * (1 - (2.f / 3.f) * u2 + (1.f / 5.f) * u4) + 0.5f));
}

struct CpuRef {
	int hidden; bool silu; int log2t; int enc_w; int out_w; size_t n_net;
	std::vector<float> P;  // all params as float
	std::vector<uint32_t> offs;

	void init(const std::vector<__half>& params) {
		P.resize(params.size());
		for (size_t i = 0; i < params.size(); ++i) P[i] = __half2float(params[i]);
		const float log2s = std::log2(1.5f);
		uint32_t off = 0;
		offs.clear();
		for (int l = 0; l < 16; ++l) {
			const float scale = exp2f(l * log2s) * 16 - 1.0f;
			const uint32_t res = (uint32_t)ceilf(scale) + 1;
			uint64_t dense = (uint64_t)res * res * res;
			dense = (dense + 7) / 8 * 8;
			offs.push_back(off);
			off += (uint32_t)std::min<uint64_t>(dense, 1ull << log2t);
		}
		offs.push_back(off);
		if (n_net + (size_t)off * 2 != P.size()) {
			fprintf(stderr, "CPU ref: param count mismatch: net %zu + grid %u*2 != %zu\n", n_net, off, P.size());
			exit(1);
		}
	}

	void encode(const float* x, float* e) const {
		for (int i = 0; i < enc_w; ++i) e[i] = 0;
		const float log2s = std::log2(1.5f);
		const float* G = P.data() + n_net;
		for (int l = 0; l < 16; ++l) {
			const float scale = exp2f(l * log2s) * 16 - 1.0f;
			const uint32_t res = (uint32_t)ceilf(scale) + 1;
			const uint32_t size = offs[l + 1] - offs[l];
			float pos[3]; uint32_t pg[3];
			for (int d = 0; d < 3; ++d) {
				float p = fmaf(scale, x[d], 0.5f);
				float t = floorf(p);
				pg[d] = (uint32_t)(int)t;
				pos[d] = p - t;
			}
			float acc[2] = {0, 0};
			for (int c = 0; c < 8; ++c) {
				float w = 1; uint32_t q[3];
				for (int d = 0; d < 3; ++d) {
					if (c & (1 << d)) { w *= pos[d]; q[d] = pg[d] + 1; } else { w *= 1 - pos[d]; q[d] = pg[d]; }
				}
				uint32_t stride = 1, idx = 0;
				if (res <= 0x659) {
					for (int d = 0; d < 3; ++d) { idx += q[d] * stride; stride *= res; }
				} else stride = 0xFFFFFFFFu;
				if (size < stride) idx = q[0] * 1u ^ q[1] * 2654435761u ^ q[2] * 805459861u;
				idx %= size;
				const float* v = G + ((size_t)offs[l] + idx) * 2;
				acc[0] += w * v[0];
				acc[1] += w * v[1];
			}
			e[l * 2] = h(acc[0]);
			e[l * 2 + 1] = h(acc[1]);
		}
		for (int d = 0; d < 5; ++d) {  // one-blob, 4 bins, periodic
			const float xv = x[3 + d];
			auto cdf = [&](float b) { return quartic_cdf(b - xv, 4.f) + quartic_cdf(b - xv - 1.f, 4.f) + quartic_cdf(b - xv + 1.f, 4.f); };
			for (int k = 0; k < 4; ++k) e[32 + d * 4 + k] = h(cdf((k + 1) / 4.f) - cdf(k / 4.f));
		}
		for (int d = 0; d < 6; ++d) e[52 + d] = h(x[8 + d]);
		for (int d = 58; d < enc_w; ++d) e[d] = 1.0f;  // tcnn pads the encoding with 1 (acts as a bias input)
	}

	void eval(const float* x, float* rgb) const {
		std::vector<float> a(enc_w), b(64);
		encode(x, a.data());
		const float* W = P.data();
		int in_w = enc_w;
		for (int layer = 0; layer < hidden; ++layer) {
			for (int o = 0; o < 64; ++o) {
				float s = 0;
				for (int i = 0; i < in_w; ++i) s += W[o * in_w + i] * a[i];
				s = silu ? s / (1 + expf(-s)) : std::max(s, 0.f);
				b[o] = h(s);
			}
			W += 64 * in_w;
			in_w = 64;
			a.assign(b.begin(), b.end());
		}
		for (int c = 0; c < 3; ++c) {
			float s = 0;
			for (int i = 0; i < 64; ++i) s += W[c * 64 + i] * a[i];
			rgb[c] = std::max(h(s), 0.f);
		}
	}
};

int main(int argc, char** argv) {
	int hidden = 5, log2t = 19, iters = 1000, device = 0;
	std::string act = "ReLU", tag = "default", engine = "tcnn";
	int fused_wg = 0, grid_levels = -1;
	bool bench_only = false, no_bench = false, clean = false;
	double lr = 1e-2, ema = 0.99;
	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		if (a == "--hidden") hidden = atoi(argv[++i]);
		else if (a == "--act") act = argv[++i];
		else if (a == "--log2t") log2t = atoi(argv[++i]);
		else if (a == "--iters") iters = atoi(argv[++i]);
		else if (a == "--bench-only") bench_only = true;
		else if (a == "--no-bench") no_bench = true;
		else if (a == "--tag") tag = argv[++i];
		else if (a == "--lr") lr = atof(argv[++i]);
		else if (a == "--ema") ema = atof(argv[++i]);
		else if (a == "--clean") clean = true;
		else if (a == "--device") device = atoi(argv[++i]);
		else if (a == "--engine") engine = argv[++i];
		else if (a == "--fused-wg") fused_wg = atoi(argv[++i]);
		else if (a == "--grid-levels") grid_levels = atoi(argv[++i]);
		else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
	}

	// Cycles-like context: primary context via the driver API, non-blocking stream.
	CU(cuInit(0));
	CUdevice dev; CU(cuDeviceGet(&dev, device));
	char name[256]; CU(cuDeviceGetName(name, sizeof(name), dev));
	CUcontext ctx; CU(cuDevicePrimaryCtxRetain(&ctx, dev));
	CU(cuCtxPushCurrent(ctx));
	CUstream stream; CU(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
	printf("device %d: %s  tag=%s hidden=%d act=%s log2t=%d lr=%g ema=%g%s\n", device, name, tag.c_str(), hidden, act.c_str(), log2t, lr, ema, clean ? " clean-targets" : "");

	char cfg[2048];
	snprintf(cfg, sizeof(cfg),
	         "{\"engine\": \"%s\", \"network\": {\"otype\": \"FullyFusedMLP\", \"activation\": \"%s\", \"output_activation\": \"None\", "
	         "\"n_neurons\": 64, \"n_hidden_layers\": %d},"
	         " \"encoding\": {\"otype\": \"Composite\", \"nested\": ["
	         "{\"n_dims_to_encode\": 3, \"otype\": \"HashGrid\", \"n_levels\": 16, \"n_features_per_level\": 2,"
	         " \"log2_hashmap_size\": %d, \"base_resolution\": 16, \"per_level_scale\": 1.5},"
	         "{\"n_dims_to_encode\": 5, \"otype\": \"OneBlob\", \"n_bins\": 4}, {\"otype\": \"Identity\"}]},"
	         " \"optimizer\": {\"otype\": \"EMA\", \"decay\": %g, \"nested\": {\"otype\": \"Adam\", \"learning_rate\": %g,"
	         " \"beta1\": 0.9, \"beta2\": 0.99, \"epsilon\": 1e-15, \"l2_reg\": 1e-6}},"
	         " \"nrc\": {\"warmup_infer_batch\": 2097152, \"warmup_train_batch\": 2097152, \"fused_wg\": %d, \"fused_grid_levels\": %d}}",
	         engine.c_str(), act.c_str(), hidden, log2t, ema, lr, fused_wg, grid_levels);

	size_t free0, total; CK(cudaMemGetInfo(&free0, &total));
	auto t0 = std::chrono::steady_clock::now();
	NrcEngine* e = nrc_create(cfg, (void*)stream);
	if (!e) { fprintf(stderr, "nrc_create failed\n"); return 1; }
	const double t_create = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	size_t free1; CK(cudaMemGetInfo(&free1, &total));
	int n_net, enc_w, out_w; nrc_debug_layout(e, &n_net, &enc_w, &out_w);
	const int n_params = nrc_debug_params(e, 1, nullptr, 0);
	printf("create %.2f s, device memory %.0f MB (warm-up 2^21), params %d (network %d), encoding width %d, output width %d\n",
	       t_create, (free0 - free1) / 1e6, n_params, n_net, enc_w, out_w);
	const std::string used = nrc_debug_engine(e);
	const bool fused = used != "tcnn";
	printf("engine requested %s, in use: %s\n", engine.c_str(), used.c_str());
	if (engine != "tcnn" && !fused) { printf("ENGINE FAIL: fused engine not in use\n"); }

	// Record pool on the device (as Cycles kernels would write it).
	const size_t POOL = 1 << 21;
	std::vector<__half> h_in; std::vector<float> h_clean, h_noisy;
	make_records(POOL, 1, h_in, h_clean, h_noisy);
	void *d_in, *d_t, *d_out;
	CK(cudaMalloc(&d_in, POOL * NRC_IN * sizeof(__half)));
	CK(cudaMalloc(&d_t, POOL * 3 * sizeof(float)));
	CK(cudaMalloc(&d_out, POOL * 3 * sizeof(float)));
	CK(cudaMemcpy(d_in, h_in.data(), POOL * NRC_IN * sizeof(__half), cudaMemcpyHostToDevice));
	CK(cudaMemcpy(d_t, (clean ? h_clean : h_noisy).data(), POOL * 3 * sizeof(float), cudaMemcpyHostToDevice));
	// Held-out set.
	const size_t NE = 1 << 16;
	std::vector<__half> e_in; std::vector<float> e_clean, e_noisy;
	make_records(NE, 2, e_in, e_clean, e_noisy);
	void* d_ein; CK(cudaMalloc(&d_ein, NE * NRC_IN * sizeof(__half)));
	CK(cudaMemcpy(d_ein, e_in.data(), NE * NRC_IN * sizeof(__half), cudaMemcpyHostToDevice));
	std::vector<float> pred(NE * 3);
	auto evaluate = [&]() {
		if (nrc_infer(e, (int)NE, d_ein, (float*)d_out, (void*)stream)) exit(1);
		CU(cuStreamSynchronize(stream));
		CK(cudaMemcpy(pred.data(), d_out, NE * 3 * sizeof(float), cudaMemcpyDeviceToHost));
		return rel_mse(pred, e_clean, NE);
	};

	int fails = 0;
	if (!bench_only) {
		const double noise = rel_mse(e_noisy, e_clean, NE);
		const double l0 = evaluate();
		printf("train: held-out relMSE vs clean field: init %.4f, one-sample noise level %.4f\n", l0, noise);
		std::mt19937 rng(3);
		const int BATCH = 65536, STEPS = 4;  // NRC recipe: 4 Adam steps of 16384 records per frame
		double l = l0;
		auto tt = std::chrono::steady_clock::now();
		for (int it = 1; it <= iters; ++it) {
			const size_t off = (rng() % ((POOL - BATCH) / 256)) * 256;
			if (nrc_train(e, BATCH, (const __half*)d_in + off * NRC_IN, (const float*)d_t + off * 3, STEPS, (void*)stream)) return 1;
			if (it % (iters / 10) == 0 || it == 10 || it == 50) {
				l = evaluate();
				printf("  iter %5d (%6d steps): relMSE %.5f\n", it, it * STEPS, l);
			}
		}
		const double tsec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tt).count();
		const bool ok = l < 0.25 * l0;  // loss falls 4x; the one-sample noise level is reported, not required
		printf("TRAIN %s: relMSE %.4f -> %.5f (%.1fx lower; noise level %.4f), %.2f s incl. evals\n", ok ? "PASS" : "FAIL",
		       l0, l, l0 / l, noise, tsec);
		fails += !ok;

		// CPU reference on 4096 held-out queries, EMA weights.
		std::vector<__half> params(n_params);
		nrc_debug_params(e, 1, params.data(), params.size());
		CpuRef ref{hidden, act == "SiLU", log2t, enc_w, out_w, (size_t)n_net};
		if (enc_w != 64) { printf("CPU ref expects encoding width 64, got %d\n", enc_w); return 1; }
		ref.init(params);
		evaluate();
		{  // encoding alone: GPU vs CPU, per segment (grid 0..31, one-blob 32..51, identity 52..57, pad 58..63)
			const int NQ = 512;
			std::vector<float> ge((size_t)NQ * enc_w), ce(enc_w);
			nrc_debug_encode(e, NQ, d_ein, ge.data(), (void*)stream);
			double seg[4] = {0, 0, 0, 0};
			for (int i = 0; i < NQ; ++i) {
				float x[14];
				for (int d = 0; d < 14; ++d) x[d] = __half2float(e_in[i * NRC_IN + d]);
				ref.encode(x, ce.data());
				for (int d = 0; d < enc_w; ++d) {
					const int sgi = d < 32 ? 0 : d < 52 ? 1 : d < 58 ? 2 : 3;
					seg[sgi] = std::max(seg[sgi], (double)fabs(ge[(size_t)i * enc_w + d] - ce[d]));
				}
			}
			printf("ENC max |gpu-cpu|: grid %.3g, one-blob %.3g, identity %.3g, pad %.3g\n", seg[0], seg[1], seg[2], seg[3]);
		}
		double max_abs = 0, max_rel = 0, sum_rel = 0; int bad = 0; const int NC = 4096;
		for (int i = 0; i < NC; ++i) {
			float x[14], r[3];
			for (int d = 0; d < 14; ++d) x[d] = __half2float(e_in[i * NRC_IN + d]);
			ref.eval(x, r);
			for (int c = 0; c < 3; ++c) {
				const double d = fabs(pred[i * 3 + c] - r[c]);
				const double rel = d / (fabs(r[c]) + 1e-2);
				max_abs = std::max(max_abs, d); max_rel = std::max(max_rel, rel); sum_rel += rel;
				bad += d > 0.01 * fabs(r[c]) + 0.01;  // fp16 tolerance: 1% relative + 0.01 absolute
			}
		}
		printf("CPUREF %s: %d queries, max |gpu-cpu| %.4g, max rel %.4g, mean rel %.3g, %d/%d outside 1%%+0.01\n",
		       bad == 0 ? "PASS" : "FAIL", NC, max_abs, max_rel, sum_rel / (NC * 3), bad, NC * 3);
		fails += bad != 0;
	}

	if (fused) {  // fused engine vs tcnn on the same EMA weights (trained unless --bench-only), held-out queries
		std::vector<float> pf(NE * 3), pt(NE * 3);
		void* d_out2; CK(cudaMalloc(&d_out2, NE * 3 * sizeof(float)));
		if (nrc_infer(e, (int)NE, d_ein, (float*)d_out, (void*)stream) ||
		    nrc_debug_infer_tcnn(e, (int)NE, d_ein, (float*)d_out2, (void*)stream)) return 1;
		CU(cuStreamSynchronize(stream));
		CK(cudaMemcpy(pf.data(), d_out, NE * 3 * sizeof(float), cudaMemcpyDeviceToHost));
		CK(cudaMemcpy(pt.data(), d_out2, NE * 3 * sizeof(float), cudaMemcpyDeviceToHost));
		CK(cudaFree(d_out2));
		double max_abs = 0, max_rel = 0, sum_rel = 0, max_ref = 0; int bad = 0;
		for (size_t i = 0; i < NE * 3; ++i) {
			const double d = fabs((double)pf[i] - pt[i]);
			const double rel = d / (fabs(pt[i]) + 1e-2);
			max_abs = std::max(max_abs, d); max_rel = std::max(max_rel, rel); sum_rel += rel;
			max_ref = std::max(max_ref, (double)fabs(pt[i]));
			bad += !(d <= 0.01 * fabs(pt[i]) + 0.01);  // fp16 tolerance: 1% relative + 0.01 absolute; NaN is bad
		}
		printf("XCHECK %s: %s vs tcnn, %zu queries (%s weights), max |d| %.4g, max rel %.4g, mean rel %.3g, max |tcnn| %.3g, "
		       "%d/%zu outside 1%%+0.01\n", bad == 0 ? "PASS" : "FAIL", used.c_str(), NE, bench_only ? "initial" : "trained",
		       max_abs, max_rel, sum_rel / (NE * 3), max_ref, bad, NE * 3);
		fails += bad != 0;
	} else if (engine != "tcnn") {
		++fails;
	}

	if (!no_bench) {
		cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
		auto timeit = [&](auto fn, int reps) {
			fn(); fn();
			CK(cudaEventRecord(a, (cudaStream_t)stream));
			for (int r = 0; r < reps; ++r) fn();
			CK(cudaEventRecord(b, (cudaStream_t)stream));
			CK(cudaEventSynchronize(b));
			float ms; CK(cudaEventElapsedTime(&ms, a, b));
			return ms / reps;
		};
		for (int lg = 16; lg <= 21; ++lg) {
			const int n = 1 << lg;
			const float mi = timeit([&] { nrc_infer(e, n, d_in, (float*)d_out, (void*)stream); }, 20);
			const float mt = timeit([&] { nrc_train(e, n, d_in, (const float*)d_t, 1, (void*)stream); }, 10);
			const float mtc = fused ? timeit([&] { nrc_debug_infer_tcnn(e, n, d_in, (float*)d_out, (void*)stream); }, 20) : mi;
			printf("BENCH {\"tag\":\"%s\",\"gpu\":\"%s\",\"engine\":\"%s\",\"hidden\":%d,\"act\":\"%s\",\"log2t\":%d,\"batch\":%d,"
			       "\"infer_ms\":%.4f,\"infer_qps\":%.4g,\"infer_ns_per_query\":%.4f,\"tcnn_infer_ms\":%.4f,"
			       "\"tcnn_ns_per_query\":%.4f,\"speedup\":%.3f,\"train_ms\":%.4f,\"train_rps\":%.4g}\n",
			       tag.c_str(), name, used.c_str(), hidden, act.c_str(), log2t, n, mi, n / (mi * 1e-3), mi * 1e6 / n, mtc,
			       mtc * 1e6 / n, mtc / mi, mt, n / (mt * 1e-3));
		}
		const float mr = timeit([&] { nrc_train(e, 65536, d_in, (const float*)d_t, 4, (void*)stream); }, 20);
		printf("BENCH {\"tag\":\"%s\",\"recipe\":\"4x16384\",\"train_call_ms\":%.4f}\n", tag.c_str(), mr);
		fflush(stdout);
	}

	nrc_destroy(e);
	CU(cuStreamDestroy(stream));
	CU(cuCtxPopCurrent(&ctx));
	CU(cuDevicePrimaryCtxRelease(dev));
	printf("%s\n", fails ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
	return fails ? 1 : 0;
}
