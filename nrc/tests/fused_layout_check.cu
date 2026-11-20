// fused_layout_check: host-only check of the conventions nrc_fused.cu hard-codes for wgmma, against CuTe's SM90 GMMA
// definitions (CUTLASS 3.5.1 in tcnn). The wgmma path cannot run on an A40, so this is its only pre-H100 check.
//   1. accumulator fragment of m64n64 / m64n16 (CLayout): value 4j + 2r + c of thread 32w + 4g + tq is
//      (row 16w + g + 8r, col 8j + 2tq + c), the mma.sync m16n8 convention per warp;
//   2. register A fragment of m64k16 (ALayout): value 4s + 2r + c is (row 16w + g + 8r, k 8s + 2tq + c), so the
//      epilogue's pairs (d[8kb + 2i], d[8kb + 2i + 1]) are A register i of k slice kb;
//   3. shared-memory tiles (Layout_K_SW128_Atom<half_t> tiled to 64x64 and 16x64): element (r, k) at swz(r, k);
//   4. smem descriptor fields CuTe's make_gmma_desc derives for those tiles (type B128, SBO 64, LBO 1) and the
//      32-byte step per k16 slice, which desc_sw128 + 2 kb encodes.
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>

#include <cstdio>
#include <cstdlib>

using namespace cute;

static int fails = 0;
#define EXPECT(cond, ...) do { if (!(cond)) { if (fails < 20) printf(__VA_ARGS__); ++fails; } } while (0)

static uint32_t swz(uint32_t r, uint32_t k) { return r * 64 + ((((k >> 3) ^ (r & 7)) << 3) | (k & 7)); }

template <class CL>
static void check_frag(CL cl, int nvals, const char* name) {
	for (int t = 0; t < 128; ++t) {
		const int w = t / 32, g = (t % 32) / 4, tq = t % 4;
		for (int v = 0; v < nvals; ++v) {
			const int c = v & 1, r = (v >> 1) & 1, j = v >> 2;
			const int idx = cl(t, v);  // column-major (M=64, N)
			const int m = idx % 64, n = idx / 64;
			EXPECT(m == 16 * w + g + 8 * r && n == 8 * j + 2 * tq + c, "%s: t %d v %d -> (%d,%d)\n", name, t, v, m, n);
		}
	}
}

template <class Shp>
static void check_tile(Shp shp, int rows, const char* name) {
	auto layout = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{}, shp);
	half_t* buf = (half_t*)aligned_alloc(1024, 64 * 64 * sizeof(half_t));
	Tensor s = make_tensor(make_smem_ptr(buf), layout);
	for (int r = 0; r < rows; ++r)
		for (int k = 0; k < 64; ++k) {
			const long off = (long)(raw_pointer_cast(&s(r, k)) - buf);
			EXPECT(off == (long)swz(r, k), "%s: (%d,%d) at %ld, swz %u\n", name, r, k, off, swz(r, k));
		}
	free(buf);
}

// Descriptor fields as CuTe's make_gmma_desc<Major::K> derives them (its constexpr part; the start address needs a
// device-only cast that does not compile in this host test): layout type, LBO and SBO in 16-byte units, plus the
// byte offset of each k16 slice of the partitioned tile. nrc_fused.cu uses type 1 (B128), LBO 1, SBO 64, +32 B per slice.
template <bool OPERAND_A, class MMA, class Shp>
static void check_desc(MMA mma, Shp shp, const char* name) {
	half_t* buf = (half_t*)aligned_alloc(1024, 64 * 64 * sizeof(half_t));
	Tensor s = make_tensor(make_smem_ptr(buf), tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{}, shp));
	auto thr = mma.get_thread_slice(0);
	auto check = [&](auto p) {
		Tensor k0 = flatten(p(_, 0, 0));  // (rows, 16) as make_gmma_desc sees it
		Tensor u = recast<uint128_t const>(k0);
		constexpr GMMA::LayoutType lt = GMMA::layout_type(u);
		Layout canon = logical_divide(layout(u), make_tile(Layout<_8, _1>{}, Layout<_2, _1>{}));
		const int sbo = int(stride<0, 1>(canon)), lbo = int(stride<1, 0>(canon));
		EXPECT(lt == GMMA::LayoutType::B128 && sbo == 64 && lbo == 1, "%s: type %d SBO %d LBO %d\n", name, (int)lt, sbo, lbo);
		for (int kb = 0; kb < 4; ++kb) {
			const long off = (long)((char*)raw_pointer_cast(p(_, 0, kb).data()) - (char*)buf);
			EXPECT(off == 32 * kb, "%s: k slice %d at byte %ld\n", name, kb, off);
		}
	};
	if constexpr (OPERAND_A) check(thr.partition_A(s));
	else check(thr.partition_B(s));
	free(buf);
}

int main() {
	check_frag(GMMA::CLayout_64x64{}, 32, "CLayout_64x64");
	check_frag(GMMA::CLayout_64x16{}, 8, "CLayout_64x16");
	check_frag(GMMA::ALayout_64x16{}, 8, "ALayout_64x16");
	check_tile(Shape<_64, _64>{}, 64, "tile 64x64");
	check_tile(Shape<_16, _64>{}, 16, "tile 16x64");

	check_desc<true>(make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{}), Shape<_64, _64>{}, "SS64 A");
	check_desc<false>(make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{}), Shape<_64, _64>{}, "SS64 B");
	check_desc<false>(make_tiled_mma(SM90_64x64x16_F32F16F16_RS<GMMA::Major::K, GMMA::Major::K>{}), Shape<_64, _64>{}, "RS64 B");
	check_desc<false>(make_tiled_mma(SM90_64x16x16_F32F16F16_RS<GMMA::Major::K, GMMA::Major::K>{}), Shape<_16, _64>{}, "RS16 B");
	printf("LAYOUT %s (%d mismatches)\n", fails ? "FAIL" : "PASS", fails);
	return fails ? 1 : 0;
}
