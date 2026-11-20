#include "cascade.cuh"

// LFM3 cascade FFN on CUDA. T = 1 only (the recipe is used on single-token steps).
//
//   CASCADE_UP      sel(g, top-c) -> gathered up dots -> sel(act, top-k within c)
//   CASCADE_DOWN    gathered down dots with a unit-major int8 repack and per-32-block f16 scales
//   CASCADE_EXPAND  compact logits [V', 1] -> full vocab [V, 1] with a -inf sentinel
//
// The selection is exact: positive IEEE-754 floats are monotone in their bit pattern, so bisecting
// [bits(min), bits(max)] 32 times finds the exact c-th (k-th) largest value. Both selections run in
// one block: F and c are small enough for the scores to live in shared memory.

#define CASCADE_SEL_BLOCK 1024
#define CASCADE_SEL_MAX   8192   // scores held in shared memory (32 KB)
#define CASCADE_UP_BLOCK  256
#define CASCADE_DOWN_SLICES 8

static __device__ __forceinline__ int block_reduce_min(int v, int * smem) {
    smem[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] = min(smem[threadIdx.x], smem[threadIdx.x + s]);
        }
        __syncthreads();
    }
    const int r = smem[0];
    __syncthreads();
    return r;
}

static __device__ __forceinline__ int block_reduce_max(int v, int * smem) {
    smem[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] = max(smem[threadIdx.x], smem[threadIdx.x + s]);
        }
        __syncthreads();
    }
    const int r = smem[0];
    __syncthreads();
    return r;
}

static __device__ __forceinline__ int block_reduce_sum(int v, int * smem) {
    smem[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    const int r = smem[0];
    __syncthreads();
    return r;
}

// exact top-`target` of a score array, compacted into an ascending list of unit ids.
// thread t owns the contiguous slice [t*per, (t+1)*per) so a scan of per-thread counts yields an
// ascending list with no sort (same property the Metal kernel relies on).
// stage 0 scores |silu(g_j)| over j < F; stage 1 scores |act[list_in[i]]| over the c candidates.
template<bool STAGE0>
__global__ void cascade_sel_kernel(
        const float * __restrict__ g,
        const float * __restrict__ act,
        const int   * __restrict__ list_in,
        float       * __restrict__ list_out,
        const int F,
        const int c,
        const int nin,
        const int target) {
    extern __shared__ float s[];              // nin scores
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    const int per = (nin + nt - 1) / nt;

    for (int i = 0; i < per; ++i) {
        const int t = tid*per + i;
        if (t < nin) {
            if (STAGE0) {
                const float gv = g[t];
                s[t] = fabsf(gv / (1.0f + expf(-gv)));
            } else {
                s[t] = fabsf(act[list_in[t]]);
            }
        } else {
            s[t] = 0.0f;
        }
    }
    __syncthreads();

    int lo = 0x7fffffff, hi = 0;
    for (int i = 0; i < per; ++i) {
        const int t = tid*per + i;
        if (t < nin) {
            const int b = __float_as_int(s[t]);
            lo = min(lo, b);
            hi = max(hi, b);
        }
    }
    __shared__ int scr[CASCADE_SEL_BLOCK];
    lo = block_reduce_min(lo, scr);
    hi = block_reduce_max(hi, scr);

    for (int it = 0; it < 32; ++it) {
        const int   mid = lo + (hi - lo + 1)/2;
        const float thr = __int_as_float(mid);
        int cnt = 0;
        for (int i = 0; i < per; ++i) {
            const int t = tid*per + i;
            if (t < nin && s[t] >= thr) {
                ++cnt;
            }
        }
        const int tot = block_reduce_sum(cnt, scr);
        if (tot >= target) { lo = mid; } else { hi = mid - 1; }
        __syncthreads();
    }
    const float thr = __int_as_float(lo);

    int cnt = 0;
    for (int i = 0; i < per; ++i) {
        const int t = tid*per + i;
        if (t < nin && s[t] >= thr && s[t] > 0.0f) {
            ++cnt;
        }
    }
    // exclusive scan of cnt over threads (Hillis-Steele)
    scr[tid] = cnt;
    __syncthreads();
    for (int off = 1; off < nt; off <<= 1) {
        const int v = (tid >= off) ? scr[tid - off] : 0;
        __syncthreads();
        scr[tid] += v;
        __syncthreads();
    }
    const int base = scr[tid] - cnt;
    int pos = base;
    for (int i = 0; i < per; ++i) {
        const int t = tid*per + i;
        if (t < nin && s[t] >= thr && s[t] > 0.0f) {
            list_out[pos++] = STAGE0 ? (float) t : (float) list_in[t];
        }
    }

    // the full count of written entries is the last element of the inclusive scan
    __syncthreads();
    const int total = scr[nt - 1];
    // pad with a valid unit id: a short list would otherwise leave uninitialised ids that the next
    // stage reads as unit indices
    for (int p = total + tid; p < target; p += nt) {
        list_out[p] = STAGE0 ? 0.0f : (list_in[0] >= 0.0f ? list_in[0] : 0.0f);
    }
}

// one block per candidate: dot of the unit's int8 row against x, with the per-32-block scales
__global__ void cascade_up_kernel(
        const char  * __restrict__ wu,     // [F, D] int8, unit-major
        const __half* __restrict__ wus,    // [F, D/32] f16 scales
        const float * __restrict__ x,      // [D]
        const float * __restrict__ g,      // [F]
        const float * __restrict__ list,   // [c] candidate unit ids
        float       * __restrict__ act,    // [F]
        const int D,
        const int c) {
    const int idx = blockIdx.x;
    if (idx >= c) {
        return;
    }
    const int u   = min(max((int) list[idx], 0), F - 1);
    const int NB  = D / 32;
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;

    __shared__ float red[CASCADE_UP_BLOCK];

    const char *   row = wu  + (size_t) u*D;
    const __half * wsr = wus + (size_t) u*NB;

    float acc = 0.0f;
    // thread t owns the values [t*per, (t+1)*per); per is 8 for D = 2048 / 256 threads
    const int per = D / nt;
    for (int i = 0; i < per; i += 4) {
        const int o = tid*per + i;
        // 4 consecutive int8 values, all inside one 32-value block when per | 32
        const int blk = o >> 5;
        const float sc = __half2float(wsr[blk]);
        float p = 0.0f;
        for (int j = 0; j < 4; ++j) {
            p += x[o + j] * (float) row[o + j];
        }
        acc += p * sc;
    }

    red[tid] = acc;
    __syncthreads();
    for (int st = nt/2; st > 0; st >>= 1) {
        if (tid < st) {
            red[tid] += red[tid + st];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const float gv = g[u];
        act[u] = (gv / (1.0f + expf(-gv))) * red[0];
    }
}

// one block per 128 outputs x one slice of the unit list; partials summed by cascade_down_reduce
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
    const int chunk = 128;
    const int nch   = D / chunk;
    const int sl    = blockIdx.x % nslice;
    const int ch    = blockIdx.x / nslice;
    const int o     = ch*chunk + threadIdx.x;      // output index
    if (o >= D) {
        return;
    }
    const int per = (k + nslice - 1) / nslice;
    const int n0  = sl*per;
    const int n1  = min(n0 + per, k);

    float acc = 0.0f;
    int cur_b = -1;
    float abl = 0.0f;
    for (int n = n0; n < n1; ++n) {
        const int u = min(max((int) list[n], 0), (int) F - 1);
        const int b = u >> 5;
        if (b != cur_b) {
            if (cur_b >= 0) {
                acc += abl * __half2float(st[(size_t) cur_b*D + o]);
            }
            cur_b = b;
            abl = 0.0f;
        }
        abl += act[u] * (float) wt[(size_t) u*D + o];
    }
    if (cur_b >= 0) {
        acc += abl * __half2float(st[(size_t) cur_b*D + o]);
    }
    part[(size_t) sl*D + o] = acc;
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

// CASCADE_CUDA_SKIP=sel0|up|sel1|down|reduce disables one stage at a time, for bisecting a fault
static bool cascade_skip(const char * what) {
    static const char * e = getenv("CASCADE_CUDA_SKIP");
    return e != nullptr && strstr(e, what) != nullptr;
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
    const int shmem = (int) (F*sizeof(float));

    if (!cascade_skip("sel0"))
    cascade_sel_kernel<true><<<1, CASCADE_SEL_BLOCK, shmem, st>>>(
        (const float *) g->data, act, (const int *) lsk, lst, F, c, F, c);
    CUDA_CHECK(cudaGetLastError());

    if (!cascade_skip("up"))
    cascade_up_kernel<<<c, CASCADE_UP_BLOCK, 0, st>>>(
        (const char *) wu->data, (const __half *) wus->data,
        (const float *) x->data, (const float *) g->data, lst, act, D, c);
    CUDA_CHECK(cudaGetLastError());

    if (!cascade_skip("sel1"))
    cascade_sel_kernel<false><<<1, CASCADE_SEL_BLOCK, (int) (c*sizeof(float)), st>>>(
        (const float *) g->data, act, (const int *) lst, lsk, F, c, c, k);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_cascade_down(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * sel  = dst->src[0];
    const ggml_tensor * wt   = dst->src[2];
    const ggml_tensor * st   = dst->src[3];
    const ggml_tensor * sp   = dst->src[4];

    const int D = (int) dst->ne[0];
    const int k = ggml_get_op_params_i32(dst, 0);
    const int S = CASCADE_DOWN_SLICES;

    const float * act  = (const float *) sel->data;
    const float * list = (const float *) sel->data + 2*sel->ne[0];

    cudaStream_t s = ctx.stream();
    const int F = (int) sel->ne[0];
    if (!cascade_skip("down"))
    cascade_down_kernel<<<(D/128)*S, 128, 0, s>>>(
        (const char *) wt->data, (const __half *) st->data, list, act,
        (float *) sp->data, D, F, k, S);
    CUDA_CHECK(cudaGetLastError());

    if (!cascade_skip("reduce"))
    cascade_down_reduce_kernel<<<(D + 255)/256, 256, 0, s>>>(
        (const float *) sp->data, (float *) dst->data, D, S);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_cascade_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * lt   = dst->src[0];
    const ggml_tensor * inv  = dst->src[1];
    const ggml_tensor * ninf = dst->src[2];

    const int V  = (int) dst->ne[0];
    const int T  = (int) dst->ne[1];
    const int Vt = (int) lt->ne[0];

    cascade_expand_kernel<<<(V*T + 255)/256, 256, 0, ctx.stream()>>>(
        (const float *) lt->data, (const int *) inv->data, (const float *) ninf->data,
        (float *) dst->data, V, T, Vt);
    CUDA_CHECK(cudaGetLastError());
}
