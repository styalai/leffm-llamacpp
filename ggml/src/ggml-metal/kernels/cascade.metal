#include "common.h"

// Cascade FFN, stage 2 (LFM3): y[o] = sum_j a_j * w_down[o, j] over the selected units.
//
// `sel` is the buffer the selection stage produces: row 1 holds the selected unit ids as f32,
// ordered by 32-unit block, and row 2 holds their activations. Weights come from the host-side
// repack, never decoded on the GPU:
//   wt[chunk][unit][128]  int8   - one 128-byte line per (output chunk, unit)
//   st[chunk][block][128] float  - the original per-(output, unit-block) scale
// One threadgroup per output chunk, one output per thread: the threadgroup reads exactly one
// 128-byte line per selected unit, coalesced, and applies the block scale once per 32 units.
kernel void kernel_cascade_down(
        constant ggml_metal_kargs_cascade_down & args,
        device const char  * wt,
        device const float * st,
        device const float * act,
        device const float * list,     // [k] selected unit ids, ascending
        device       float * y,
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint ntg[[threads_per_threadgroup]]) {
    if (args.ablate) {
        return;
    }
    const int F  = args.F;
    const int K  = args.k;
    const int ch = (int) tgpig;

    device const char4  * W = (device const char4  *) (wt + (size_t) ch * F * 128);
    device const float4 * S = (device const float4 *) (st + (size_t) ch * (F/32) * 128);

    const int sg   = (int) (tpitg / 32);
    const int lane = (int) (tpitg % 32);
    const int nsg  = (int) (ntg / 32);

    // Even split of the ascending list, so every simdgroup gets the same number of units (block
    // ownership would match the scale reads to the block but load-balances badly: block occupancy
    // varies 0..32, and the threadgroup barrier waits on the slowest warp).
    const int per_sg = (K + nsg - 1) / nsg;
    const int n0 = sg*per_sg;
    const int n1 = min(n0 + per_sg, K);

    // LLAMA_CASCADE_DOWN_PREFIX: read units 0..K-1 instead of the selected list. The result is
    // wrong on purpose -- it turns every chunk into a sequential stream, which isolates "the
    // gathered access granularity is the limiter" from every other explanation.
    const bool prefix = args.prefix != 0;

    threadgroup float4 part[32][32];

    float4 acc  = float4(0.0f);
    float4 ablk = float4(0.0f);
    int cur_b = -1;

    int n = n0;
    while (n < n1) {
        if (n + 8 <= n1) {
            // eight ids: two vector loads for the ids, eight independent line loads in flight
            const float4 ia = *(device const float4 *) (list + n);
            const float4 ib = *(device const float4 *) (list + n + 4);
            const int j0 = prefix ? n     : (int) ia.x;
            const int j7 = prefix ? n + 7 : (int) ib.w;
            const int b8 = j0 >> 5;
            if (b8 == (j7 >> 5)) {
                if (b8 != cur_b) {
                    if (cur_b >= 0) {
                        acc += ablk * S[(size_t) cur_b*32 + lane];
                    }
                    cur_b = b8;
                    ablk = float4(0.0f);
                }
                const int j[8] = { j0,
                                   prefix ? n + 1 : (int) ia.y,
                                   prefix ? n + 2 : (int) ia.z,
                                   prefix ? n + 3 : (int) ia.w,
                                   prefix ? n + 4 : (int) ib.x,
                                   prefix ? n + 5 : (int) ib.y,
                                   prefix ? n + 6 : (int) ib.z,
                                   j7 };
                char4 w[8];
                for (int u = 0; u < 8; ++u) {
                    w[u] = W[(size_t) j[u]*32 + lane];
                }
                for (int u = 0; u < 8; ++u) {
                    ablk += float4(act[j[u]]) * float4(w[u]);
                }
                n += 8;
                continue;
            }
        }
        if (n + 4 <= n1) {
            const float4 ids = *(device const float4 *) (list + n);
            const int j0 = (int) ids.x;
            const int j3 = (int) ids.w;
            const int b4 = j0 >> 5;
            if (b4 == (j3 >> 5)) {
                if (b4 != cur_b) {
                    if (cur_b >= 0) {
                        acc += ablk * S[(size_t) cur_b*32 + lane];
                    }
                    cur_b = b4;
                    ablk = float4(0.0f);
                }
                const char4 w0 = W[(size_t) j0*32 + lane];
                const char4 w1 = W[(size_t)(int) ids.y*32 + lane];
                const char4 w2 = W[(size_t)(int) ids.z*32 + lane];
                const char4 w3 = W[(size_t) j3*32 + lane];
                ablk += float4(act[j0]) * float4(w0)
                      + float4(act[(int) ids.y]) * float4(w1)
                      + float4(act[(int) ids.z]) * float4(w2)
                      + float4(act[j3]) * float4(w3);
                n += 4;
                continue;
            }
        }
        // a group that straddles a block boundary, or the tail
        {
            const int j = (int) list[n];
            const int b = j >> 5;
            if (b != cur_b) {
                if (cur_b >= 0) {
                    acc += ablk * S[(size_t) cur_b*32 + lane];
                }
                cur_b = b;
                ablk = float4(0.0f);
            }
            ablk += float4(act[j]) * float4(W[(size_t) j*32 + lane]);
            ++n;
        }
    }
    if (cur_b >= 0) {
        acc += ablk * S[(size_t) cur_b*32 + lane];
    }

    part[sg][lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float4 sum = float4(0.0f);
        for (int t = 0; t < nsg; ++t) {
            sum += part[t][lane];
        }
        ((device float4 *) y)[(size_t) ch*32 + lane] = sum;
    }
}

// Exact top-c / top-k selection in a single threadgroup: positive floats are monotone in their bit
// pattern, so 32 halvings over [bits(min), bits(max)] find the exact n-th largest value. Scores live
// in registers; every pass reduces a count across the threadgroup (simdgroup sums + one shared
// reduction), so no atomics and no grid-wide sync are needed.
kernel void kernel_cascade_sel(
        constant ggml_metal_kargs_cascade_sel & args,
        device const float * g,        // [F] gate pre-activations (stage 0)
        device const float * act,      // [F] activations         (stage 1)
        device const uchar * mask1,    // [F] stage-1 selection   (stage 1)
        device       float * list,     // [k] ascending unit ids  (stage 1 output)
        uint tpitg[[thread_position_in_threadgroup]],
        uint ntg[[threads_per_threadgroup]]) {
    const int F   = args.F;
    const int per = (F + (int) ntg - 1) / (int) ntg;

    float s[32];
    int   js[32];
    for (int i = 0; i < per; ++i) {
        // contiguous slices: thread t owns [t*per, (t+1)*per), so a prefix scan yields an
        // ascending unit list with no sort
        const int j = (int) tpitg * per + i;
        js[i] = j;
        float v = 0.0f;
        if (j < F) {
            if (args.stage == 0) {
                const float gv = g[j];
                v = fabs(gv / (1.0f + exp(-gv)));
            } else if (mask1[j] != 0) {
                v = fabs(act[j]);
            }
        }
        s[i] = v;
    }

    threadgroup int tl[64];
    threadgroup int th[64];

    const uint lane = tpitg % 32;
    const uint wid  = tpitg / 32;
    const uint nw   = ((uint) ntg + 31) / 32;

    int lo = 0x7fffffff, hi = 0;
    for (int i = 0; i < per; ++i) {
        const int b = as_type<int>(s[i]);
        lo = min(lo, b);
        hi = max(hi, b);
    }
    lo = simd_min(lo);
    hi = simd_max(hi);
    if (lane == 0) { tl[wid] = lo; th[wid] = hi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tpitg == 0) {
        int a = 0x7fffffff, b = 0;
        for (uint w = 0; w < nw; ++w) { a = min(a, tl[w]); b = max(b, th[w]); }
        tl[0] = a; th[0] = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lo = tl[0]; hi = th[0];

    for (int it = 0; it < 32; ++it) {
        const int   mid = lo + (hi - lo + 1)/2;
        const float t   = as_type<float>(mid);
        int cnt = 0;
        for (int i = 0; i < per; ++i) {
            cnt += (s[i] >= t) ? 1 : 0;
        }
        cnt = simd_sum(cnt);
        if (lane == 0) { tl[wid] = cnt; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tpitg == 0) {
            int tot = 0;
            for (uint w = 0; w < nw; ++w) { tot += tl[w]; }
            tl[0] = tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int tot = tl[0];
        if (tot >= args.c) { lo = mid; } else { hi = mid - 1; }
        if (tpitg == 0) { tl[1] = lo; th[1] = hi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        lo = tl[1]; hi = th[1];
    }

    const float thr = as_type<float>(lo);

    // compaction: per-thread counts, simdgroup prefix, one small threadgroup scan, then an
    // ascending write. Only stage 1 needs the list; stage 0 writes the mask the up stage reads.
    if (args.stage == 0) {
        for (int i = 0; i < per; ++i) {
            const int j = js[i];
            if (j < F) {
                ((device uchar *) list)[j] = (s[i] >= thr && s[i] > 0.0f) ? 1 : 0;
            }
        }
        return;
    }

    int cnt = 0;
    for (int i = 0; i < per; ++i) {
        cnt += (s[i] >= thr && s[i] > 0.0f) ? 1 : 0;
    }
    const int inclusive = simd_prefix_inclusive_sum(cnt);
    threadgroup int wsum[16];
    const uint wid2 = tpitg / 32;
    const uint nw2  = (ntg + 31) / 32;
    if (lane == 31) {
        wsum[wid2] = inclusive;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tpitg == 0) {
        int run = 0;
        for (uint w = 0; w < nw2; ++w) {
            const int t = wsum[w];
            wsum[w] = run;
            run += t;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int pos = wsum[wid2] + inclusive - cnt;
    for (int i = 0; i < per; ++i) {
        if (s[i] >= thr && s[i] > 0.0f) {
            list[pos++] = (float) js[i];
        }
    }
}

// u_j = x . w_up[j] for the selected units, then a_j = silu(g_j) * u_j, one simdgroup per unit.
kernel void kernel_cascade_up(
        constant ggml_metal_kargs_cascade_up & args,
        device const char  * wu,       // int8 unit-major [F, D]
        device const float * wus,      // per-32-block scales [F, D/32]
        device const float * x,        // [D]
        device const float * g,        // [F]
        device const uchar * mask,     // [F]
        device       float * act,      // [F]
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint ntg[[threads_per_threadgroup]]) {
    if (args.ablate) {
        return;
    }
    const int F  = args.F;
    const int D  = args.D;
    const int NB = args.NB;   // scale blocks per unit (D/32)

    const int per_tg = (int) (ntg / 32);
    const int j      = (int) (tgpig * per_tg + tpitg/32);
    if (j >= F || mask[j] == 0) {
        return;
    }
    const uint lane = tpitg % 32;

    // int8 unit-major row: one lane per 32-value block, so a lane reads 32 contiguous bytes
    // (8 x char4) and applies one scale. D/32 == 64 blocks give every lane two of them, and the
    // two blocks' loads are interleaved so both are in flight.
    device const char  * wr  = wu  + (size_t) j*D;
    device const float * wsr = wus + (size_t) j*NB;

    float acc = 0.0f;
    for (int b = (int) lane; b < NB; b += 32) {
        const int  b2  = b + 32;
        const bool two = (b2 < NB);
        // four 8-value chunk sums per 32-value block, each scaled by the block's d before it is
        // accumulated -- the same granularity kernel_mul_mv_q8_0_f32 uses, so the gathered dot
        // tracks the dense matmul as closely as the fp32 accumulation order allows
        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
        float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
        for (int t = 0; t < 32; t += 4) {
            const char4  v  = *(device const char4  *) (wr + b*32 + t);
            const float4 xv = *(device const float4 *) (x  + b*32 + t);
            const float  pv = xv.x*(float) v.x + xv.y*(float) v.y + xv.z*(float) v.z + xv.w*(float) v.w;
            if      (t <  8) { p0 += pv; }
            else if (t < 16) { p1 += pv; }
            else if (t < 24) { p2 += pv; }
            else             { p3 += pv; }
            if (two) {
                const char4  u  = *(device const char4  *) (wr + b2*32 + t);
                const float4 yv = *(device const float4 *) (x  + b2*32 + t);
                const float  qv = yv.x*(float) u.x + yv.y*(float) u.y + yv.z*(float) u.z + yv.w*(float) u.w;
                if      (t <  8) { q0 += qv; }
                else if (t < 16) { q1 += qv; }
                else if (t < 24) { q2 += qv; }
                else             { q3 += qv; }
            }
        }
        const float db = wsr[b];
        acc += (p0 + p1 + p2 + p3) * db;
        if (two) {
            const float db2 = wsr[b2];
            acc += (q0 + q1 + q2 + q3) * db2;
        }
    }
    acc = simd_sum(acc);

    if (lane == 0) {
        const float gv = g[j];
        act[j] = (gv / (1.0f + exp(-gv))) * acc;
    }
}
