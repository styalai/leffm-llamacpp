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
        device const char  * wt,       // [F, D] int8, one contiguous 2 KB row per unit
        device const half  * st,       // [F/32, D] f16 scales, one contiguous row per block
        device const float * act,      // [F]
        device const float * list,     // [k] selected unit ids, ascending
        device       float * part,     // [T, D] partial sums, one row per threadgroup
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint ntg[[threads_per_threadgroup]]) {
    if (args.ablate) {
        return;
    }
    const int F = args.F;
    const int K = args.k;
    const int D = args.D;
    const int T = args.T;

    // Even split of the ascending list by unit count. Work is balanced (unlike a block-owned
    // split), and a block cut between two threadgroups is harmless because the block scale
    // multiplies each partial sum linearly.
    const int per_tg = (K + T - 1) / T;
    const int n0 = (int) tgpig * per_tg;
    const int n1 = min(n0 + per_tg, K);
    if (n0 >= n1) {
        return;
    }

    device float * out = part + (size_t) tgpig * D;

    // Each thread owns four outputs. The whole threadgroup covers D in one step, so a unit's full
    // row is read as one 2 KB burst by the threadgroup (16 x 128 B lines) instead of each
    // threadgroup reading a 128 B granule of the same unit.
    for (int o = 4*(int) tpitg; o < D; o += 4*(int) ntg) {
        float4 acc  = float4(0.0f);
        float4 ablk = float4(0.0f);
        int cur_b = -1;

        int n = n0;
        while (n < n1) {
            if (n + 8 <= n1) {
                const float4 ia = *(device const float4 *) (list + n);
                const float4 ib = *(device const float4 *) (list + n + 4);
                const int j0 = (int) ia.x;
                const int j7 = (int) ib.w;
                const int b  = j0 >> 5;
                if (b == (j7 >> 5)) {
                    if (b != cur_b) {
                        if (cur_b >= 0) {
                            const half4 h = *(device const half4 *) (st + (size_t) cur_b*D + o);
                            acc += ablk * float4((float) h.x, (float) h.y, (float) h.z, (float) h.w);
                        }
                        cur_b = b;
                        ablk  = float4(0.0f);
                    }
                    const int j[8] = { j0, (int) ia.y, (int) ia.z, (int) ia.w,
                                       (int) ib.x, (int) ib.y, (int) ib.z, j7 };
                    char4 w[8];
                    for (int u = 0; u < 8; ++u) {
                        w[u] = *(device const char4 *) (wt + (size_t) j[u]*D + o);
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
                const int j[4] = { (int) ids.x, (int) ids.y, (int) ids.z, (int) ids.w };
                if ((j[0] >> 5) == (j[3] >> 5)) {
                    const int b = j[0] >> 5;
                    if (b != cur_b) {
                        if (cur_b >= 0) {
                            const half4 h = *(device const half4 *) (st + (size_t) cur_b*D + o);
                            acc += ablk * float4((float) h.x, (float) h.y, (float) h.z, (float) h.w);
                        }
                        cur_b = b;
                        ablk  = float4(0.0f);
                    }
                    const char4 w0 = *(device const char4 *) (wt + (size_t) j[0]*D + o);
                    const char4 w1 = *(device const char4 *) (wt + (size_t) j[1]*D + o);
                    const char4 w2 = *(device const char4 *) (wt + (size_t) j[2]*D + o);
                    const char4 w3 = *(device const char4 *) (wt + (size_t) j[3]*D + o);
                    ablk += float4(act[j[0]]) * float4(w0)
                          + float4(act[j[1]]) * float4(w1)
                          + float4(act[j[2]]) * float4(w2)
                          + float4(act[j[3]]) * float4(w3);
                    n += 4;
                    continue;
                }
            }
            {
                const int j = (int) list[n];
                const int b = j >> 5;
                if (b != cur_b) {
                    if (cur_b >= 0) {
                        const half4 h = *(device const half4 *) (st + (size_t) cur_b*D + o);
                        acc += ablk * float4((float) h.x, (float) h.y, (float) h.z, (float) h.w);
                    }
                    cur_b = b;
                    ablk  = float4(0.0f);
                }
                const char4 w = *(device const char4 *) (wt + (size_t) j*D + o);
                ablk += float4(act[j]) * float4(w);
                ++n;
            }
        }
        if (cur_b >= 0) {
            const half4 h = *(device const half4 *) (st + (size_t) cur_b*D + o);
            acc += ablk * float4((float) h.x, (float) h.y, (float) h.z, (float) h.w);
        }
        *(device float4 *) (out + o) = acc;
    }
}

kernel void kernel_cascade_down_reduce(
        constant ggml_metal_kargs_cascade_reduce & args,
        device const float * part,     // [T, D]
        device       float * y,        // [D]
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint ntg[[threads_per_threadgroup]]) {
    const int D = args.D;
    const int T = args.T;
    const int o = (int) tgpig * (int) ntg + (int) tpitg;
    if (o >= D) {
        return;
    }
    float s = 0.0f;
    for (int t = 0; t < T; ++t) {
        s += part[(size_t) t*D + o];
    }
    y[o] = s;
}

// Exact top-c / top-k selection in a single threadgroup: positive floats are monotone in their bit
// pattern, so 32 halvings over [bits(min), bits(max)] find the exact n-th largest value. Scores live
// in registers; every pass reduces a count across the threadgroup (simdgroup sums + one shared
// reduction), so no atomics and no grid-wide sync are needed.
kernel void kernel_cascade_sel(
        constant ggml_metal_kargs_cascade_sel & args,
        device const float * g,        // [F] gate pre-activations (stage 0 scores)
        device const float * act,      // [F] activations         (stage 1 scores)
        device const float * list_in,  // [c] stage-0 candidates  (stage 1 input)
        device       float * list_out, // ascending selected unit ids
        uint tpitg[[thread_position_in_threadgroup]],
        uint ntg[[threads_per_threadgroup]]) {
    const int  F   = args.F;
    const bool st0 = (args.stage == 0);
    // the candidate list is compact, so both stages score exactly the entries they must inspect:
    // F for the first, c for the second
    const int  nin = args.nin;
    const int  per = (nin + (int) ntg - 1) / (int) ntg;

    float s[32];
    int   js[32];
    for (int i = 0; i < per; ++i) {
        // contiguous slices: thread t owns [t*per, (t+1)*per), so a prefix scan yields an
        // ascending unit list with no sort
        const int t = (int) tpitg * per + i;
        s[i]  = 0.0f;
        js[i] = 0;
        if (t < nin) {
            const int j = st0 ? t : (int) list_in[t];
            js[i] = j;
            if (st0) {
                const float gv = g[j];
                s[i] = fabs(gv / (1.0f + exp(-gv)));
            } else {
                s[i] = fabs(act[j]);
            }
        }
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
        if (tot >= args.target) { lo = mid; } else { hi = mid - 1; }
        if (tpitg == 0) { tl[1] = lo; th[1] = hi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        lo = tl[1]; hi = th[1];
    }

    const float thr = as_type<float>(lo);

    // compaction: per-thread counts, simdgroup prefix, one small threadgroup scan, then an
    // ascending write
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
            list_out[pos++] = (float) js[i];
        }
    }
}

// u_j = x . w_up[j] for the selected units, then a_j = silu(g_j) * u_j, one simdgroup per unit.
kernel void kernel_cascade_up(
        constant ggml_metal_kargs_cascade_up & args,
        device const char  * wu,       // int8 unit-major [F, D]
        device const half  * wus,      // per-32-block scales [F, D/32], fp16 as in the source
        device const float * x,        // [D]
        device const float * g,        // [F]
        device const float * list,     // [c] candidate unit ids, ascending
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

    // Two candidate units per simdgroup: 2 x 4 KB rows interleaved gives 32 weight loads in flight
    // per lane step instead of 16, and the 32-value x block is loaded once and reused by both.
    const int warp = (int) (tpitg / 32);
    const int lane = (int) (tpitg % 32);
    const int per_tg = (int) (ntg / 32) * 2;
    const int j0 = (int) tgpig * per_tg + warp * 2;
    const int j1 = j0 + 1;

    // the candidates are a compact ascending list, so two per simdgroup are always live
    const bool a0 = j0 < args.c;
    const bool a1 = j1 < args.c;
    if (!a0 && !a1) {
        return;
    }
    const int u0 = a0 ? (args.prefix ? j0 : (int) list[j0]) : 0;
    const int u1 = a1 ? (args.prefix ? j1 : (int) list[j1]) : 0;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    if (a0 && a1) {
        device const char  * wr0 = wu  + (size_t) u0*D;
        device const char  * wr1 = wu  + (size_t) u1*D;
        device const half  * ws0 = wus + (size_t) u0*NB;
        device const half  * ws1 = wus + (size_t) u1*NB;
        for (int b = (int) lane; b < NB; b += 32) {
            float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
            float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
            // 8-byte weight loads (uint2 = 8 int8 values) with the same 8-value chunk split
            for (int t8 = 0; t8 < 4; ++t8) {
                const int t = t8*8;
                const uint2 vv = *(device const uint2 *) (wr0 + b*32 + t);
                const uint2 ww = *(device const uint2 *) (wr1 + b*32 + t);
                const float4 xa = *(device const float4 *) (x + b*32 + t);
                const float4 xb = *(device const float4 *) (x + b*32 + t + 4);
                const char4 v0 = as_type<char4>(vv.x);
                const char4 v1 = as_type<char4>(vv.y);
                const char4 w0 = as_type<char4>(ww.x);
                const char4 w1 = as_type<char4>(ww.y);
                const float pv = xa.x*(float) v0.x + xa.y*(float) v0.y + xa.z*(float) v0.z + xa.w*(float) v0.w
                               + xb.x*(float) v1.x + xb.y*(float) v1.y + xb.z*(float) v1.z + xb.w*(float) v1.w;
                const float qv = xa.x*(float) w0.x + xa.y*(float) w0.y + xa.z*(float) w0.z + xa.w*(float) w0.w
                               + xb.x*(float) w1.x + xb.y*(float) w1.y + xb.z*(float) w1.z + xb.w*(float) w1.w;
                if      (t8 == 0) { p0 += pv; q0 += qv; }
                else if (t8 == 1) { p1 += pv; q1 += qv; }
                else if (t8 == 2) { p2 += pv; q2 += qv; }
                else              { p3 += pv; q3 += qv; }
            }
            acc0 += (p0 + p1 + p2 + p3) * (float) ws0[b];
            acc1 += (q0 + q1 + q2 + q3) * (float) ws1[b];
        }
    } else {
        const int j = a0 ? u0 : u1;
        device const char  * wr = wu  + (size_t) j*D;
        device const half  * ws = wus + (size_t) j*NB;
        for (int b = (int) lane; b < NB; b += 32) {
            float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
            for (int t8 = 0; t8 < 4; ++t8) {
                const int t = t8*8;
                const uint2 vv = *(device const uint2 *) (wr + b*32 + t);
                const float4 xa = *(device const float4 *) (x + b*32 + t);
                const float4 xb = *(device const float4 *) (x + b*32 + t + 4);
                const char4 v0 = as_type<char4>(vv.x);
                const char4 v1 = as_type<char4>(vv.y);
                const float pv = xa.x*(float) v0.x + xa.y*(float) v0.y + xa.z*(float) v0.z + xa.w*(float) v0.w
                               + xb.x*(float) v1.x + xb.y*(float) v1.y + xb.z*(float) v1.z + xb.w*(float) v1.w;
                if      (t8 == 0) { p0 += pv; }
                else if (t8 == 1) { p1 += pv; }
                else if (t8 == 2) { p2 += pv; }
                else              { p3 += pv; }
            }
            acc0 += (p0 + p1 + p2 + p3) * (float) ws[b];
        }
    }

    // warp-collective: every lane must participate, so reduce before the lane-0 store
    const float s0 = simd_sum(acc0);
    const float s1 = simd_sum(acc1);

    if (lane == 0) {
        if (a0) {
            const float gv = g[u0];
            act[u0] = (gv / (1.0f + exp(-gv))) * s0;
        }
        if (a1) {
            const float gv = g[u1];
            act[u1] = (gv / (1.0f + exp(-gv))) * s1;
        }
    }
}

kernel void kernel_cascade_expand(
        constant ggml_metal_kargs_cascade_expand & args,
        device const float * lt,       // [V', T] compact logits
        device const int   * inv,      // [V] position of each vocab id, or V' for dropped
        device const float * ninf,     // [1] the -inf sentinel
        device       float * out,      // [V, T]
        uint gid [[thread_position_in_grid]]) {
    const int V  = args.V;
    const int T  = args.T;
    const int Vt = args.Vt;
    const int n  = V*T;
    if ((int) gid >= n) {
        return;
    }
    const int v = (int) gid % V;
    const int t = (int) gid / V;
    const int p = inv[v];
    out[gid] = (p < Vt) ? lt[p + t*Vt] : ninf[0];
}
