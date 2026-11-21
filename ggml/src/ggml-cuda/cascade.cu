#include "cascade.cuh"

#include <cstdlib>
#include <cstring>

// LFM3 cascade FFN on CUDA. T = 1 only (the recipe is used on single-token steps).
//
//   CASCADE_UP      sel(g, top-c) -> gathered up dots -> sel(act, top-k within c)
//   CASCADE_DOWN    gathered down dots with a unit-major int8 repack and per-32-block f16 scales
//   CASCADE_EXPAND  compact logits [V', 1] -> full vocab [V, 1] with a -inf sentinel
//
// The selection is exact: positive IEEE-754 floats are monotone in their bit pattern, so four 8-bit
// radix passes find the exact c-th (k-th) largest value. Both selections run in one block: F and c
// are small enough for the scores to live in shared memory.

#define CASCADE_SEL_BLOCK 1024
#define CASCADE_SEL_MAX   8192   // scores held in shared memory (32 KB)
#define CASCADE_UP_BLOCK  128   // four warps, one candidate each
#define CASCADE_UP_WARPS   4
#define CASCADE_DOWN_SLICES 32

// exact top-`target` of a non-negative score array, compacted into an ascending list of unit ids.
// thread t owns the contiguous slice [t*per, (t+1)*per) so a scan of per-thread counts yields an
// ascending list with no sort (same property the Metal kernel relies on).
// stage 0 scores |silu(g_j)| over j < F; stage 1 scores |act[list_in[i]]| over the c candidates.
//
// Selection is a radix select on the IEEE-754 bit pattern of the (non-negative) score: positive
// floats are monotone in their bit pattern, so four 8-bit histogram passes find the exact
// target-th largest value. It replaces a 32-round bisection whose per-round block-wide reduction
// dominated the cascade on an L4: measured with CASCADE_CUDA_SKIP, the two selections cost 6.6 of
// the 7.0 ms/token the recipe lost against dense, i.e. ~94% of the whole regression.
template<bool STAGE0>
__global__ void cascade_sel_kernel(
        const float * __restrict__ g,
        const float * __restrict__ act,
        const float * __restrict__ list_in,
        float       * __restrict__ list_out,
        const int F,
        const int c,
        const int nin,
        const int target,
        const int npass) {
    extern __shared__ float s[];              // nin scores
    __shared__ int hist[256];
    __shared__ int scan[CASCADE_SEL_BLOCK];
    __shared__ int sel_digit, sel_need;
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    const int per = (nin + nt - 1) / nt;

    // the block covers nt*per >= nin slots and only [0, nin) exists in shared memory: the padding
    // must not be written (every later read is guarded by t < nin)
    for (int i = 0; i < per; ++i) {
        const int t = tid*per + i;
        if (t < nin) {
            if (STAGE0) {
                const float gv = g[t];
                s[i*nt + tid] = fabsf(gv / (1.0f + __expf(-gv)));
            } else {
                s[i*nt + tid] = fabsf(act[min(max((int) list_in[t], 0), F - 1)]);
            }
        }
    }
    __syncthreads();

    // four 8-bit passes: prefix holds the bits already resolved, `need` is the rank still sought
    // inside that prefix bucket
    unsigned prefix = 0u;
    int need = target;
    for (int sh = 24; sh >= 0; sh -= 8) {
        if ((24 - sh)/8 >= npass) {
            break;      // timing only: fewer passes leave the threshold unresolved
        }
        for (int b = tid; b < 256; b += nt) {
            hist[b] = 0;
        }
        __syncthreads();
        const unsigned above = (sh == 24) ? 0u : (0xFFFFFFFFu << (sh + 8));
        const unsigned want  = prefix & above;
        for (int i = 0; i < per; ++i) {
            const int t = tid*per + i;
            if (t < nin) {
                const unsigned bits = __float_as_uint(s[i*nt + tid]);
                if ((bits & above) == want) {
                    atomicAdd(&hist[(bits >> sh) & 0xFF], 1);
                }
            }
        }
        __syncthreads();
        // pick the digit: the smallest d whose bucket holds the rank we still seek, i.e. the
        // smallest d with (count of items in digits above d) < need. One warp does it with
        // shuffles: a block-wide scan of 256 bins cost 17 barriers per pass.
        if (tid < 32) {
            const int lo = 8*tid;
            const int t0 = hist[lo + 0], t1 = hist[lo + 1], t2 = hist[lo + 2], t3 = hist[lo + 3];
            const int t4 = hist[lo + 4], t5 = hist[lo + 5], t6 = hist[lo + 6], t7 = hist[lo + 7];
            const int tot = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7;
            int above = tot;                       // inclusive suffix over the lanes
            for (int off = 1; off < 32; off <<= 1) {
                const int v = __shfl_down_sync(0xFFFFFFFFu, above, off);
                if (tid + off < 32) {
                    above += v;
                }
            }
            const int rank = need - (above - tot);  // rank inside this lane's eight bins
            if (rank > 0 && rank <= tot) {
                const int h[8] = {t0, t1, t2, t3, t4, t5, t6, t7};
                int cum       = 0;
                int digit     = lo + 7;
                int rest      = rank;
                bool found    = false;
                #pragma unroll
                for (int d = 7; d >= 0; --d) {
                    if (!found && cum + h[d] >= rank) {   // first hit from the top holds the rank
                        digit = lo + d;
                        rest  = rank - cum;
                        found = true;
                    }
                    cum += h[d];
                }
                sel_digit = digit;
                sel_need  = rest;
            }
        }
        __syncthreads();
        prefix |= (unsigned) sel_digit << sh;
        need = sel_need;
        __syncthreads();
    }
    const unsigned thr = prefix;
    if (npass < 4) {
        return;         // timing only: skip the compaction, the output list stays stale
    }

    // compaction: ascending ids of the qualifying entries, capped at `target` so the list is exactly
    // the size the gathered kernels index
    int cnt = 0;
    for (int i = 0; i < per; ++i) {
        const int t = tid*per + i;
        if (t < nin && __float_as_uint(s[i*nt + tid]) >= thr) {
            ++cnt;
        }
    }
    scan[tid] = cnt;
    __syncthreads();
    for (int off = 1; off < nt; off <<= 1) {          // Hillis-Steele exclusive scan
        const int v = (tid >= off) ? scan[tid - off] : 0;
        __syncthreads();
        scan[tid] += v;
        __syncthreads();
    }
    int pos = scan[tid] - cnt;
    for (int i = 0; i < per; ++i) {
        const int t = tid*per + i;
        if (t < nin && __float_as_uint(s[i*nt + tid]) >= thr && pos < target) {
            list_out[pos++] = STAGE0 ? (float) t : list_in[t];
        }
    }
}

// one warp per candidate: dot of the unit's int8 row against x, with the per-32-block scales.
// The row is fetched as 16-byte loads (16 values never straddle a 32-value scale block, and the row
// is 16-byte aligned), and the dot reduces through warp shuffles, so a block of four warps has no
// barrier at all: the earlier block-wide reduction cost more than the 2 KB a candidate actually reads.
__global__ void cascade_up_kernel(
        const char  * __restrict__ wu,     // [F, D] int8, unit-major
        const __half* __restrict__ wus,    // [F, D/32] f16 scales
        const float * __restrict__ x,      // [D]
        const float * __restrict__ g,      // [F]
        const float * __restrict__ list,   // [c] candidate unit ids
        float       * __restrict__ act,    // [F]
        const int D,
        const int F,
        const int c) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int idx  = blockIdx.x*CASCADE_UP_WARPS + warp;
    if (idx >= c) {
        return;
    }
    const int u = min(max((int) list[idx], 0), F - 1);
    const char *   row = wu  + (size_t) u*D;
    const __half * wsr = wus + (size_t) u*(D/32);

    const int chunks = D / 16;
    float acc = 0.0f;
    for (int k = lane; k < chunks; k += 32) {
        const int o = k*16;
        const int4   w  = *(const int4   *) (row + o);
        const float4 x0 = *(const float4 *) (x + o);
        const float4 x1 = *(const float4 *) (x + o + 4);
        const float4 x2 = *(const float4 *) (x + o + 8);
        const float4 x3 = *(const float4 *) (x + o + 12);
        float p = 0.0f;
        p += x0.x*(float)(signed char)( w.x        & 0xFF);
        p += x0.y*(float)(signed char)((w.x >>  8) & 0xFF);
        p += x0.z*(float)(signed char)((w.x >> 16) & 0xFF);
        p += x0.w*(float)(signed char)((w.x >> 24) & 0xFF);
        p += x1.x*(float)(signed char)( w.y        & 0xFF);
        p += x1.y*(float)(signed char)((w.y >>  8) & 0xFF);
        p += x1.z*(float)(signed char)((w.y >> 16) & 0xFF);
        p += x1.w*(float)(signed char)((w.y >> 24) & 0xFF);
        p += x2.x*(float)(signed char)( w.z        & 0xFF);
        p += x2.y*(float)(signed char)((w.z >>  8) & 0xFF);
        p += x2.z*(float)(signed char)((w.z >> 16) & 0xFF);
        p += x2.w*(float)(signed char)((w.z >> 24) & 0xFF);
        p += x3.x*(float)(signed char)( w.w        & 0xFF);
        p += x3.y*(float)(signed char)((w.w >>  8) & 0xFF);
        p += x3.z*(float)(signed char)((w.w >> 16) & 0xFF);
        p += x3.w*(float)(signed char)((w.w >> 24) & 0xFF);
        acc += p * __half2float(wsr[o >> 5]);
    }
    if (lane == 0) {
        for (int j = chunks*16; j < D; ++j) {     // D is a multiple of 16 in practice
            acc += x[j] * (float) row[j] * __half2float(wsr[j >> 5]);
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        const float gv = g[u];
        act[u] = (gv / (1.0f + expf(-gv))) * acc;
    }
}

// one accumulated unit for one output column group: adds act[u] * wt[u*D + o .. +3] to the running
// partials, flushing them into the result whenever the unit's 32-unit scale block changes
static __device__ __forceinline__ void cascade_down_step(
        const int u, const int v, const int D, const int o,
        const __half * __restrict__ st, const float * __restrict__ act,
        int & cur_b, float & a0, float & a1, float & a2, float & a3,
        float & p0, float & p1, float & p2, float & p3) {
    const int b = u >> 5;
    if (b != cur_b) {
        if (cur_b >= 0) {
            const __half * sp = st + (size_t) cur_b*D + o;
            a0 += p0 * __half2float(sp[0]);
            a1 += p1 * __half2float(sp[1]);
            a2 += p2 * __half2float(sp[2]);
            a3 += p3 * __half2float(sp[3]);
        }
        cur_b = b;
        p0 = p1 = p2 = p3 = 0.0f;
    }
    const float a = act[u];
    p0 += a * (float) (signed char) ( v        & 0xFF);
    p1 += a * (float) (signed char) ((v >>  8) & 0xFF);
    p2 += a * (float) (signed char) ((v >> 16) & 0xFF);
    p3 += a * (float) (signed char) ((v >> 24) & 0xFF);
}

// one block per 512 outputs x one slice of the unit list; partials summed by cascade_down_reduce.
// each thread owns 4 consecutive outputs, reads them as one 32-bit load, and keeps eight units in
// flight: this gather is limited by bytes in flight, not by instruction count (it ran at 66 GB/s
// against the up projection's 180).
__global__ void cascade_down_kernel(
        const char  * __restrict__ wt,     // [F, D] int8, unit-major
        const __half* __restrict__ st,     // [F/32, D] f16 scales
        const float * __restrict__ list,   // [k] ascending unit ids
        const float * __restrict__ act,    // [F]
        float       * __restrict__ part,   // [S, D]
        const int D,
        const int F,
        const int k,
        const int nslice) {
    const int vec   = 4;
    const int chunk = 128*vec;
    const int sl    = blockIdx.x % nslice;
    const int ch    = blockIdx.x / nslice;
    const int o     = ch*chunk + threadIdx.x*vec;
    if (o + vec > D) {
        return;   // D is a multiple of 4 by the repack precondition (n_embd % 128 == 0)
    }
    const int per = (k + nslice - 1) / nslice;
    const int n0  = sl*per;
    const int n1  = min(n0 + per, k);
    const int Fm1 = F - 1;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
    int cur_b = -1;

    int n = n0;
    for (; n + 8 <= n1; n += 8) {
        const int u0 = min(max((int) list[n + 0], 0), Fm1);
        const int u1 = min(max((int) list[n + 1], 0), Fm1);
        const int u2 = min(max((int) list[n + 2], 0), Fm1);
        const int u3 = min(max((int) list[n + 3], 0), Fm1);
        const int u4 = min(max((int) list[n + 4], 0), Fm1);
        const int u5 = min(max((int) list[n + 5], 0), Fm1);
        const int u6 = min(max((int) list[n + 6], 0), Fm1);
        const int u7 = min(max((int) list[n + 7], 0), Fm1);
        const int v0 = *(const int *) (wt + (size_t) u0*D + o);
        const int v1 = *(const int *) (wt + (size_t) u1*D + o);
        const int v2 = *(const int *) (wt + (size_t) u2*D + o);
        const int v3 = *(const int *) (wt + (size_t) u3*D + o);
        const int v4 = *(const int *) (wt + (size_t) u4*D + o);
        const int v5 = *(const int *) (wt + (size_t) u5*D + o);
        const int v6 = *(const int *) (wt + (size_t) u6*D + o);
        const int v7 = *(const int *) (wt + (size_t) u7*D + o);
        cascade_down_step(u0, v0, D, o, st, act, cur_b, a0, a1, a2, a3, p0, p1, p2, p3);
        cascade_down_step(u1, v1, D, o, st, act, cur_b, a0, a1, a2, a3, p0, p1, p2, p3);
        cascade_down_step(u2, v2, D, o, st, act, cur_b, a0, a1, a2, a3, p0, p1, p2, p3);
        cascade_down_step(u3, v3, D, o, st, act, cur_b, a0, a1, a2, a3, p0, p1, p2, p3);
        cascade_down_step(u4, v4, D, o, st, act, cur_b, a0, a1, a2, a3, p0, p1, p2, p3);
        cascade_down_step(u5, v5, D, o, st, act, cur_b, a0, a1, a2, a3, p0, p1, p2, p3);
        cascade_down_step(u6, v6, D, o, st, act, cur_b, a0, a1, a2, a3, p0, p1, p2, p3);
        cascade_down_step(u7, v7, D, o, st, act, cur_b, a0, a1, a2, a3, p0, p1, p2, p3);
    }
    for (; n < n1; ++n) {
        const int u = min(max((int) list[n], 0), Fm1);
        const int v = *(const int *) (wt + (size_t) u*D + o);
        cascade_down_step(u, v, D, o, st, act, cur_b, a0, a1, a2, a3, p0, p1, p2, p3);
    }
    if (cur_b >= 0) {
        const __half * sp = st + (size_t) cur_b*D + o;
        a0 += p0 * __half2float(sp[0]);
        a1 += p1 * __half2float(sp[1]);
        a2 += p2 * __half2float(sp[2]);
        a3 += p3 * __half2float(sp[3]);
    }
    part[(size_t) sl*D + o    ] = a0;
    part[(size_t) sl*D + o + 1] = a1;
    part[(size_t) sl*D + o + 2] = a2;
    part[(size_t) sl*D + o + 3] = a3;
}

__global__ void cascade_down_reduce_kernel(
        const float * __restrict__ part,   // [S, D]
        float       * __restrict__ y,      // [D]
        const int D,
        const int nslice) {
    const int o = blockIdx.x*blockDim.x + threadIdx.x;
    if (o >= D) {
        return;
    }
    float s = 0.0f;
    for (int t = 0; t < nslice; ++t) {
        s += part[(size_t) t*D + o];
    }
    y[o] = s;
}

__global__ void cascade_expand_kernel(
        const float * __restrict__ lt,     // [V', T]
        const int   * __restrict__ inv,    // [V]
        const float * __restrict__ ninf,   // [1]
        float       * __restrict__ out,    // [V, T]
        const int V,
        const int T,
        const int Vt) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= V*T) {
        return;
    }
    const int v = i % V;
    const int t = i / V;
    const int p = inv[v];
    out[i] = (p < Vt) ? lt[p + (size_t) t*Vt] : ninf[0];
}

// CASCADE_CUDA_TIME=1 times each launch with CUDA events and prints the totals at exit. The skip
// ablation cannot separate a stage from the weight traffic it removes; this can.
struct ct_timers {
    bool on;
    cudaEvent_t e0, e1;
    double sel0 = 0, up = 0, sel1 = 0, down = 0, reduce = 0, expand = 0;
    long n_sel0 = 0, n_up = 0, n_sel1 = 0, n_down = 0, n_reduce = 0, n_expand = 0;
    ct_timers() {
        const char * e = getenv("CASCADE_CUDA_TIME");
        on = e != nullptr && e[0] == '1';
        if (on) { cudaEventCreate(&e0); cudaEventCreate(&e1); }
    }
    void dump() {
        fprintf(stderr,
                "CASCADE_CUDA_TIME sel0=%.1fms/%ld up=%.1fms/%ld sel1=%.1fms/%ld down=%.1fms/%ld "
                "reduce=%.1fms/%ld expand=%.1fms/%ld\n",
                sel0, n_sel0, up, n_up, sel1, n_sel1, down, n_down, reduce, n_reduce, expand, n_expand);
    }
    void start(cudaStream_t s) { if (on) { cudaEventRecord(e0, s); } }
    void stop(cudaStream_t s, double & slot, long & cnt) {
        if (!on) { return; }
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        slot += ms;
        cnt  += 1;
    }
};
static ct_timers g_ct;

// CASCADE_CUDA_SKIP=sel0|up|sel1|down|reduce disables one stage at a time, for bisecting a fault
static bool cascade_skip(const char * what) {
    static const char * e = getenv("CASCADE_CUDA_SKIP");
    return e != nullptr && strstr(e, what) != nullptr;
}

static int sel_npass() {
    static int v = -1;
    if (v < 0) {
        v = 4;
        const char * e = getenv("CASCADE_CUDA_SELPHASE");
        if (e != nullptr) {
            const int t = atoi(e);
            if (t >= 0 && t < 4) {
                v = t;
            }
        }
    }
    return v;
}

void ggml_cuda_op_cascade_up(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * g   = dst->src[0];
    const ggml_tensor * x   = dst->src[1];
    const ggml_tensor * wu  = dst->src[3];
    const ggml_tensor * wus = dst->src[4];

    const int F = (int) g->ne[0];
    const int D = (int) x->ne[0];
    const int c = ggml_get_op_params_i32(dst, 0);
    const int k = ggml_get_op_params_i32(dst, 1);

    GGML_ASSERT(F <= CASCADE_SEL_MAX && c <= F && k <= c);

    float * sel = (float *) dst->data;
    float * act = sel;                        // row 0
    float * lst = sel + F;                    // row 1 (stage-0 list)
    float * lsk = sel + 2*F;                  // row 2 (stage-1 list)

    cudaStream_t st = ctx.stream();
    // the scores live in a strided layout (s[i*nt + tid]) and address the full nt*per slots, which is
    // at least nin, so the buffer is sized by slots and not by nin
    const int per0  = (F + CASCADE_SEL_BLOCK - 1)/CASCADE_SEL_BLOCK;
    const int shmem = (int) (per0*CASCADE_SEL_BLOCK*sizeof(float));

    g_ct.start(st);
    if (!cascade_skip("sel0"))
    cascade_sel_kernel<true><<<1, CASCADE_SEL_BLOCK, shmem, st>>>(
        (const float *) g->data, act, lsk, lst, F, c, F, c, sel_npass());
    g_ct.stop(st, g_ct.sel0, g_ct.n_sel0);
    CUDA_CHECK(cudaGetLastError());

    g_ct.start(st);
    if (!cascade_skip("up"))
    cascade_up_kernel<<<(c + CASCADE_UP_WARPS - 1)/CASCADE_UP_WARPS, CASCADE_UP_BLOCK, 0, st>>>(
        (const char *) wu->data, (const __half *) wus->data,
        (const float *) x->data, (const float *) g->data, lst, act, D, F, c);
    g_ct.stop(st, g_ct.up, g_ct.n_up);
    CUDA_CHECK(cudaGetLastError());

    g_ct.start(st);
    if (!cascade_skip("sel1")) {
        const int per1 = (c + CASCADE_SEL_BLOCK - 1)/CASCADE_SEL_BLOCK;
        cascade_sel_kernel<false><<<1, CASCADE_SEL_BLOCK, (int) (per1*CASCADE_SEL_BLOCK*sizeof(float)), st>>>(
            (const float *) g->data, act, lst, lsk, F, c, c, k, sel_npass());
    }
    g_ct.stop(st, g_ct.sel1, g_ct.n_sel1);
    CUDA_CHECK(cudaGetLastError());
}

// CASCADE_CUDA_SLICES overrides the down split (default CASCADE_DOWN_SLICES, max 64): the gathers are
// limited by bytes in flight, so the split is a tuning knob, and the scratch is sized for the maximum
static int cascade_down_slices() {
    static int s = -1;
    if (s < 0) {
        s = CASCADE_DOWN_SLICES;
        const char * e = getenv("CASCADE_CUDA_SLICES");
        if (e != nullptr) {
            const int v = atoi(e);
            if (v >= 1 && v <= GGML_CASCADE_DOWN_SLICES_MAX) {
                s = v;
            }
        }
    }
    return s;
}

void ggml_cuda_op_cascade_down(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * sel  = dst->src[0];
    const ggml_tensor * wt   = dst->src[2];
    const ggml_tensor * st   = dst->src[3];
    const ggml_tensor * sp   = dst->src[4];

    const int D = (int) dst->ne[0];
    const int k = ggml_get_op_params_i32(dst, 0);
    const int S = cascade_down_slices();

    const float * act  = (const float *) sel->data;
    const float * list = (const float *) sel->data + 2*sel->ne[0];

    cudaStream_t s = ctx.stream();
    const int F = (int) sel->ne[0];
    g_ct.start(s);
    if (!cascade_skip("down"))
    cascade_down_kernel<<<((D + 511)/512)*S, 128, 0, s>>>(
        (const char *) wt->data, (const __half *) st->data, list, act,
        (float *) sp->data, D, F, k, S);
    g_ct.stop(s, g_ct.down, g_ct.n_down);
    CUDA_CHECK(cudaGetLastError());

    g_ct.start(s);
    if (!cascade_skip("reduce"))
    cascade_down_reduce_kernel<<<(D + 255)/256, 256, 0, s>>>(
        (const float *) sp->data, (float *) dst->data, D, S);
    g_ct.stop(s, g_ct.reduce, g_ct.n_reduce);
    if (g_ct.on && g_ct.n_reduce % 24 == 0) {
        g_ct.dump();
        if (g_ct.n_up >= 48) {
            g_ct.on = false;    // stop before graph capture, which cannot contain a blocking sync
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_cascade_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * lt   = dst->src[0];
    const ggml_tensor * inv  = dst->src[1];
    const ggml_tensor * ninf = dst->src[2];

    const int V  = (int) dst->ne[0];
    const int T  = (int) dst->ne[1];
    const int Vt = (int) lt->ne[0];

    g_ct.start(ctx.stream());
    cascade_expand_kernel<<<(V*T + 255)/256, 256, 0, ctx.stream()>>>(
        (const float *) lt->data, (const int *) inv->data, (const float *) ninf->data,
        (float *) dst->data, V, T, Vt);
    g_ct.stop(ctx.stream(), g_ct.expand, g_ct.n_expand);
    CUDA_CHECK(cudaGetLastError());
}
