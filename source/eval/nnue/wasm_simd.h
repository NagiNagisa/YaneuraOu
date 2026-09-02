#include "../../config.h"

#if defined(USE_WASM_SIMD)

/*
	注意

	下記の affine() は、現在どこからも呼び出されていない。

	wasmビルドでは -DUSE_SSE42 も同時に指定されるため、config.hによって
	USE_SSSE3が定義され、AffineTransform / AffineTransformSparseInput の
	ReadParameters()は、SSSE3向けに重みを並び替えて(GetWeightIndexScrambled())
	読み込む。
	しかし下記の affine() は、重みを並び替えなしのA[m][n_stride]として
	読むので、これを呼び出すと評価値が正しく計算されない。

	また、SSSE3経路のほうにはblock-sparse input(AffineTransformSparseInput)の
	最適化が入っているので、速度的にもそちらのほうが速い。

	復活させるなら、GetWeightIndexScrambled()に対応させた上で、
	relaxed SIMDの i32x4.relaxed_dot_i8x16_i7x16_add_s を用いて書き直すこと。
	(ただし、それはSimd::m128_add_dpbusd_epi32()で行っているので、
	 わざわざこちらを復活させる必要はないと思う)
*/

#include <stdint.h>
#include <wasm_simd128.h>

namespace emscripten_wasm_simd {

template<int n, int m, int n_stride>
void affine(const int8_t A[m][n_stride], const uint8_t x[n], const int32_t b[m], int32_t y[m]);

} // namespace emscripten_wasm_simd
#endif
