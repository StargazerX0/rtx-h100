/* Test-only accessors of the NRC engine. Not part of the Cycles ABI (include/cycles_nrc.h). */
#ifndef CYCLES_NRC_DEBUG_H
#define CYCLES_NRC_DEBUG_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct NrcEngine NrcEngine;
/* Copies the fp16 parameters (inference = EMA weights if 1, training weights if 0) to host memory.
 * Layout: network weight matrices (row-major, out x in, first/hidden/last), then the hash-grid table.
 * host_fp16 == NULL returns the parameter count. Synchronises the device. */
int nrc_debug_params(NrcEngine *, int inference, void *host_fp16, size_t capacity);
int nrc_debug_layout(NrcEngine *, int *n_network_params, int *encoding_width, int *padded_output_width);
const char *nrc_debug_config(NrcEngine *);
/* Encoding only: n x encoding_width floats (row j = query j). Synchronises the stream. */
int nrc_debug_encode(NrcEngine *, int n, const void *d_inputs, float *host_out, void *cu_stream);
/* tcnn inference on the EMA weights regardless of the configured engine. No sync. */
int nrc_debug_infer_tcnn(NrcEngine *, int n, const void *d_inputs, float *d_rgb, void *cu_stream);
/* Engine in use: "tcnn", or "wgmma|mma wg=.. ctas_per_sm=.. grid=.. regs=.. smem=..". */
const char *nrc_debug_engine(NrcEngine *);
#ifdef __cplusplus
}
#endif
#endif
