// nrc_fused.h: fused NRC inference (hash grid + one-blob + identity encoding, then the ReLU MLP) in one persistent
// kernel. The MMA core is wgmma on sm_90a ("wgmma") or mma.sync on sm_80+ ("mma"). It reads tiny-cuda-nn's fp16
// parameter buffer as is (network weights row-major out x in, first/hidden/last, then the hash-grid table).
// Supported model: 3D hash grid, 16 levels x 2 features, linear interpolation; one-blob (4 bins) on 5 dims; identity on
// 6 dims; FullyFusedMLP, 64 neurons, ReLU, output activation None, 1-8 hidden layers, padded output width 16.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace nrc_fused {

constexpr int N_LEVELS = 16;
constexpr int MAX_HIDDEN = 8;

struct Model {
	int hidden = 5;             // n_hidden_layers (number of 64x64 matrices, the first one reads the encoding)
	uint32_t log2_hashmap = 19;
	uint32_t base_resolution = 16;
	float per_level_scale = 1.5f;
};

enum Core { CORE_MMA = 1, CORE_WGMMA = 2 };

struct Plan {
	Core core = CORE_MMA;
	int wg = 2;                 // warpgroups per CTA
	int grid = 0;               // persistent CTAs (SMs x resident CTAs per SM)
	int blocks_per_sm = 0;
	size_t smem = 0;
	int regs = 0;
	int levels_per_pass = 0;    // 0: the fused kernel gathers all 16 grid levels itself; k > 0: split mode, a grid
	                            // pass encodes k levels per sweep into a 16 x n half2 workspace first
	Model model;
	uint32_t offsets[N_LEVELS + 1];  // grid-table offsets in entries (tcnn's offset table)
	uint32_t n_net = 0;         // network parameters (fp16) before the grid table
};

// Offsets and parameter counts as tcnn computes them. Returns the number of grid entries (x 2 features).
uint32_t grid_offsets(const Model& m, uint32_t* offsets);

// core: CORE_WGMMA needs a compute capability 9.0 device. wg: 0 = pick by occupancy (2 or 4). Returns false and sets
// err on failure (unsupported model, device or launch configuration).
bool setup(Plan& p, const Model& m, Core core, int wg, const char** err);

// out[i*3 + c] = max(0, MLP(encode(in[i*16 .. i*16+13])))[c] with params = tcnn's fp16 parameter buffer.
// enc: workspace of 16 * n half2 when p.levels_per_pass > 0 (else unused).
cudaError_t infer(const Plan& p, cudaStream_t stream, uint32_t n, const __half* in, float* out, const __half* params,
                  __half2* enc);

}  // namespace nrc_fused
