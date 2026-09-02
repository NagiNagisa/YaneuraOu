#ifndef CLASSIC_SIMD_H_INCLUDED
#define CLASSIC_SIMD_H_INCLUDED

#if defined(USE_AVX2)
    #include <immintrin.h>

#elif defined(USE_SSE41)
    #include <smmintrin.h>

#elif defined(USE_SSSE3)
    #include <tmmintrin.h>

#elif defined(USE_SSE2)
    #include <emmintrin.h>

#elif defined(USE_NEON)
    #include <arm_neon.h>
#endif

#if defined(USE_WASM_RELAXED_SIMD)
    #include <wasm_simd128.h>
#endif

// CPU capability switch for NNUE dot-product code. For SFNN H1=7 networks
// (fc0: ->8, packed tail: 14->64->1), measured AVX512VNNI was slower than
// AVX512 maddubs/madd, so only that NNUE shape uses the fallback.
#if defined(USE_VNNI) && !defined(NNUE_SFNN_HIDDEN1_7)
    #define USE_NNUE_VNNI
#endif

namespace YaneuraOu {
namespace Simd
{


#if defined(USE_AVX512)

[[maybe_unused]] static int m512_hadd(__m512i sum, int bias) {
    return _mm512_reduce_add_epi32(sum) + bias;
}

[[maybe_unused]] static void m512_add_maddubs_epi32(__m512i& acc, __m512i a, __m512i b) {
    __m512i product0 = _mm512_maddubs_epi16(a, b);
    product0         = _mm512_madd_epi16(product0, _mm512_set1_epi16(1));
    acc              = _mm512_add_epi32(acc, product0);
}

[[maybe_unused]] static void m512_add_dpbusd_epi32(__m512i& acc, __m512i a, __m512i b) {
    #if defined(USE_NNUE_VNNI)
    acc = _mm512_dpbusd_epi32(acc, a, b);
    #else
    m512_add_maddubs_epi32(acc, a, b);
    #endif
}

#endif

#if defined(USE_AVX2)

[[maybe_unused]] static int m256_hadd(__m256i sum, int bias) {
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    sum128         = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_BADC));
    sum128         = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_CDAB));
    return _mm_cvtsi128_si32(sum128) + bias;
}

[[maybe_unused]] static void m256_add_maddubs_epi32(__m256i& acc, __m256i a, __m256i b) {
    __m256i product0 = _mm256_maddubs_epi16(a, b);
    product0         = _mm256_madd_epi16(product0, _mm256_set1_epi16(1));
    acc              = _mm256_add_epi32(acc, product0);
}

[[maybe_unused]] static void m256_add_dpbusd_epi32(__m256i& acc, __m256i a, __m256i b) {
    #if defined(USE_NNUE_VNNI)
    acc = _mm256_dpbusd_epi32(acc, a, b);
    #else
    m256_add_maddubs_epi32(acc, a, b);
    #endif
}

#endif

#if defined(USE_SSSE3)

[[maybe_unused]] static int m128_hadd(__m128i sum, int bias) {
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));  //_MM_PERM_BADC
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1));  //_MM_PERM_CDAB
    return _mm_cvtsi128_si32(sum) + bias;
}

[[maybe_unused]] static void m128_add_dpbusd_epi32(__m128i& acc, __m128i a, __m128i b) {
#if defined(USE_WASM_RELAXED_SIMD)
    /*
		yaneuraou.wasm

		relaxed SIMDの i32x4.relaxed_dot_i8x16_i7x16_add_s は、VNNIのvpdpbusdに
		相当する命令で、下のmaddubs + madd + add の3命令を1命令で行える。
		(wasmには_mm_maddubs_epi16に相当する命令がないので、emscriptenのSSE
		 エミュレーションでは、これはさらに複数命令に展開されている)

		この命令は、第1引数がsigned int8、第2引数がunsignedの7bit(0..127)で
		なければならない。(第2引数の最上位bitが立っているとき、その結果は
		実装依存となる)
		NNUEでは、a = ClippedReLUの出力(0..127) , b = 重み(int8)なので、
		引数を入れ替えて (b , a) の順で渡す。

		cf. https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md
	*/
    acc = (__m128i) wasm_i32x4_relaxed_dot_i8x16_i7x16_add((v128_t) b, (v128_t) a, (v128_t) acc);
#else
    __m128i product0 = _mm_maddubs_epi16(a, b);
    product0         = _mm_madd_epi16(product0, _mm_set1_epi16(1));
    acc              = _mm_add_epi32(acc, product0);
#endif
}

#endif

#if defined(USE_NEON_DOTPROD)

[[maybe_unused]] static void dotprod_m128_add_dpbusd_epi32(int32x4_t& acc, int8x16_t a, int8x16_t b) {
    acc = vdotq_s32(acc, a, b);
}
#endif

#if defined(USE_NEON)

[[maybe_unused]] static int neon_m128_reduce_add_epi32(int32x4_t s) {
    #if USE_NEON >= 8
    return vaddvq_s32(s);
    #else
    return s[0] + s[1] + s[2] + s[3];
    #endif
}

[[maybe_unused]] static int neon_m128_hadd(int32x4_t sum, int bias) {
    return neon_m128_reduce_add_epi32(sum) + bias;
}

#endif

#if USE_NEON >= 8
[[maybe_unused]] static void neon_m128_add_dpbusd_epi32(int32x4_t& acc, int8x16_t a, int8x16_t b) {

    int16x8_t product0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    int16x8_t product1 = vmull_high_s8(a, b);
    int16x8_t sum      = vpaddq_s16(product0, product1);
    acc                = vpadalq_s16(acc, sum);
}

#endif


} // namespace Simd 
} // namespace YaneuraOu

#endif // ifndef SIMD_H_INCLUDED
