"""
confusion_comparison.py  —  ECE410_project_tb_netlist
======================================================
Side-by-side confusion matrices: sklearn float64 vs HW Q6.10 LUT kernel.

HW kernel matrix is computed with Numba JIT (parallel prange).
The hw_dist_nb model includes the 2-cycle pipeline drain (Fix #10) so
all 256 feature dimensions contribute — matching the fixed RTL exactly.

Outputs:
  confusion_comparison.png  in the same directory as this script
"""

import sys, math, ctypes, warnings, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from numba import njit, prange

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Constants
# ─────────────────────────────────────────────────────────────────────────────
FRAC_BITS        = 10
SCALE            = 1 << FRAC_BITS           # 1024
DIST_WIDTH       = 20
MAX_DIST_Q       = (1 << DIST_WIDTH) - 1    # 0xFFFFF

FEATURE_DIM      = 256
FEAT_SINGLE      = 128
FEAT_10BEAT      = 64
FEAT_100RR       = 64
HALF_SINGLE      = FEAT_SINGLE  // 2
HALF_10BEAT      = FEAT_10BEAT  // 2
N_BEATS_10       = 10
N_BEATS_100      = 100
NORMAL_RR        = 308

NUM_CLASSES      = 5
MAX_SV_PER_CLASS = 50
CLASS_NAMES      = ["Normal", "PVC", "AFib", "VT", "SVT"]
DEFAULT_GAMMA    = 0.25

HORNER_COEFFS = [1024, 1024, 512, 170, 42, 8, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]
EXP_INT_LUT   = [round(math.exp(-i) * SCALE) for i in range(16)]

import os
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Numba-visible constant arrays
_EXP_LUT_NB  = np.array(EXP_INT_LUT,   dtype=np.int64)
_HORNER_NB   = np.array(HORNER_COEFFS, dtype=np.int64)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Numba-JIT HW functions
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _to_i16(v):
    """Signed int16 wrap — same semantics as ctypes.c_int16."""
    v = int(v) & 0xFFFF
    if v >= 32768:
        v -= 65536
    return v


@njit(cache=True)
def hw_dist_nb(x_q, sv_q):
    """
    Fixed-point distance accumulator with 2-cycle drain.

    Models the svm_compute_core distance_matrix pipeline exactly after Fix #10:
      diff -> diff_squared -> accumulator  (2-stage delay)
    Two drain cycles after the last valid_in flush dim[254] and dim[255].
    All 256 feature dimensions are accumulated.
    """
    acc = np.int64(0)
    prev_d   = np.int64(0)
    prev_dsq = np.int64(0)
    for k in range(256):
        xi   = _to_i16(x_q[k])
        si   = _to_i16(sv_q[k])
        diff = _to_i16(xi - si)
        acc     += prev_dsq >> 10
        prev_dsq = prev_d * prev_d
        prev_d   = diff
    # drain 1
    acc     += prev_dsq >> 10
    prev_dsq = prev_d * prev_d
    # drain 2
    acc += prev_dsq >> 10
    if acc > np.int64(0xFFFFF):
        acc = np.int64(0xFFFFF)
    return acc


@njit(cache=True)
def hw_kernel_nb(gamma_q, dist_q, exp_lut, horner_c):
    """
    Range-reduction RBF kernel in Q6.10.
    exp(-γd²) = exp(-I) × exp(-F)
      I  = integer index into exp_lut (0-15)
      F  = fractional part, Horner polynomial for exp(-F/1024)
    """
    gamma_s = _to_i16(gamma_q)
    dist_u  = dist_q & np.int64(0xFFFFF)
    P       = gamma_s * dist_u
    I       = P >> 20
    F_q     = (P >> 10) & np.int64(1023)
    if I >= 16:
        return np.int64(0)
    lut_val = exp_lut[I]
    x      = _to_i16(-F_q)
    result = _to_i16(horner_c[15])
    for n in range(14, -1, -1):
        t32    = _to_i16(x) * _to_i16(result)
        result = _to_i16(horner_c[n] + (t32 >> 10))
    horner_val = np.int64(result)
    if horner_val < 0:    horner_val = np.int64(0)
    if horner_val > 1024: horner_val = np.int64(1024)
    combined = (lut_val * horner_val) >> 10
    if combined < 0:    combined = np.int64(0)
    if combined > 1024: combined = np.int64(1024)
    return combined


@njit(parallel=True, cache=True)
def compute_kernel_matrix_nb(X_q, SV_q, gamma_q, exp_lut, horner_c):
    """
    Parallel HW kernel matrix computation — Numba prange over test samples.
    X_q:  (N, 256) int32   SV_q: (M, 256) int32
    Returns (N, M) float64  (kernel values already divided by 1024).
    """
    N = X_q.shape[0]
    M = SV_q.shape[0]
    K = np.zeros((N, M), dtype=np.float64)
    for i in prange(N):
        for j in range(M):
            dq     = hw_dist_nb(X_q[i], SV_q[j])
            K[i,j] = hw_kernel_nb(gamma_q, dq, exp_lut, horner_c) / 1024.0
    return K


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Fixed-point helpers (Python-side)
# ─────────────────────────────────────────────────────────────────────────────

def float_to_q10(f):
    return ctypes.c_uint16(int(round(f * SCALE))).value

def vecs_to_q10(X: np.ndarray) -> np.ndarray:
    """(N, D) float32 → (N, D) int32  Q6.10 encoding."""
    return np.round(X * SCALE).astype(np.int32)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Multi-scale feature extraction  (128 + 64 + 64 = 256 dims)
# ─────────────────────────────────────────────────────────────────────────────

def extract_multiscale(sig, all_beat_samples, beat_idx):
    s = int(all_beat_samples[beat_idx]); n = len(sig)
    if s < HALF_SINGLE or s + HALF_SINGLE > n:
        return None
    seg1 = sig[s - HALF_SINGLE : s + HALF_SINGLE].copy()
    pk1  = np.max(np.abs(seg1))
    if pk1 < 1e-6: return None
    seg1 = (seg1 / pk1).astype(np.float32)

    half10 = N_BEATS_10 // 2
    i0 = max(0, beat_idx - half10); i1 = min(len(all_beat_samples), beat_idx + half10)
    segs2 = []
    for bi in range(i0, i1):
        bs = int(all_beat_samples[bi])
        if bs >= HALF_10BEAT and bs + HALF_10BEAT <= n:
            seg = sig[bs - HALF_10BEAT : bs + HALF_10BEAT].astype(np.float32)
            pk2 = np.max(np.abs(seg))
            if pk2 > 1e-6: segs2.append(seg / pk2)
    if not segs2: return None
    seg2 = np.mean(segs2, axis=0).astype(np.float32)

    j0 = max(0, beat_idx - N_BEATS_100)
    rr_raw = np.diff(all_beat_samples[j0 : beat_idx + 1]).astype(np.float32)
    if len(rr_raw) < 2: return None
    rr_norm = np.clip(rr_raw / NORMAL_RR, 0.0, 2.0)
    seg3 = np.interp(np.linspace(0, 1, FEAT_100RR),
                     np.linspace(0, 1, len(rr_norm)), rr_norm).astype(np.float32)
    return np.concatenate([seg1, seg2, seg3])

# ─────────────────────────────────────────────────────────────────────────────
# 4.  MIT-BIH loader
# ─────────────────────────────────────────────────────────────────────────────
ALL_MITBIH = tuple(str(r) for r in list(range(100, 125)) + list(range(200, 235)))
_BEAT_SYMS = set("NLReEjJAaSVFQf")

def load_mitbih_beats(records=ALL_MITBIH, max_per_class=300):
    import wfdb
    BMAP = {"N": 0, "L": 0, "R": 0, "e": 0,
            "V": 1, "E": 1,
            "A": 4, "a": 4, "J": 4, "S": 4}
    beats = {i: [] for i in range(NUM_CLASSES)}
    for rec in records:
        try:
            r   = wfdb.rdrecord(rec, pn_dir="mitdb")
            ann = wfdb.rdann(rec, "atr", pn_dir="mitdb")
        except Exception as e:
            print(f"  [skip] {rec}: {e}"); continue
        sig = r.p_signal[:, 0].astype(np.float32)
        beat_samp_list, beat_sym_list = [], []
        for s, sym in zip(ann.sample, ann.symbol):
            if sym in _BEAT_SYMS:
                beat_samp_list.append(s); beat_sym_list.append(sym)
        if len(beat_samp_list) < 3: continue
        all_beat_samples = np.array(beat_samp_list, dtype=np.int32)
        afib_regions = []
        if hasattr(ann, "aux_note"):
            in_afib = False; afib_start = None
            for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note):
                if sym == "+" and aux:
                    if "(AFIB" in aux: in_afib = True; afib_start = s
                    elif in_afib and "(" in aux:
                        afib_regions.append((afib_start, s)); in_afib = False
            if in_afib and afib_start is not None:
                afib_regions.append((afib_start, len(sig)))
        for beat_idx, (s_idx, sym) in enumerate(
                zip(all_beat_samples.tolist(), beat_sym_list)):
            in_afib = any(a0 <= s_idx <= a1 for a0, a1 in afib_regions)
            if in_afib and sym in ("N", "L", "R", "V", "A"): cls = 2
            elif sym in BMAP: cls = BMAP[sym]
            else: continue
            if len(beats[cls]) >= max_per_class: continue
            feat = extract_multiscale(sig, all_beat_samples, beat_idx)
            if feat is not None: beats[cls].append(feat)
    return beats

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Synthetic ECG generators (fill gaps when real data is scarce)
# ─────────────────────────────────────────────────────────────────────────────

def _gauss(t, mu, sig, amp):
    return amp * np.exp(-0.5 * ((t - mu) / sig) ** 2)

def _rr_to_64(rr_samples):
    rr_norm = np.clip(np.asarray(rr_samples, np.float32) / NORMAL_RR, 0.0, 2.0)
    return np.interp(np.linspace(0, 1, 64),
                     np.linspace(0, 1, len(rr_norm)), rr_norm).astype(np.float32)

def _make_ms(beat_fn, rr_fn, n, noise1, noise2, rng):
    t128 = np.linspace(-1, 1, 128); t64 = np.linspace(-1, 1, 64); out = []
    for _ in range(n):
        s1 = beat_fn(t128, rng) + rng.normal(0, noise1, 128)
        s1 = (s1 / (np.max(np.abs(s1)) + 1e-9)).astype(np.float32)
        s2 = beat_fn(t64,  rng) + rng.normal(0, noise2, 64)
        s2 = (s2 / (np.max(np.abs(s2)) + 1e-9)).astype(np.float32)
        s3 = _rr_to_64(rr_fn(99, rng))
        out.append(np.concatenate([s1, s2, s3]))
    return out

def synth_normal(n, rng):
    def beat(t, r): return (_gauss(t,-0.50,0.08,0.15)+_gauss(t,-0.05,0.03,-0.15)
                            +_gauss(t, 0.00,0.04,1.00)+_gauss(t, 0.05,0.03,-0.10)
                            +_gauss(t, 0.35,0.10,0.35))
    def rr(nb, r):  return r.normal(NORMAL_RR, NORMAL_RR*0.05, nb)
    return _make_ms(beat, rr, n, 0.03, 0.02, rng)

def synth_pvc(n, rng):
    def beat(t, r):
        w = r.uniform(0.12, 0.18)
        return (_gauss(t,-0.02,w,-0.8)+_gauss(t,0.12,w*0.8,0.5)+_gauss(t,0.45,0.15,-0.4))
    def rr(nb, r):
        rv = r.normal(NORMAL_RR, NORMAL_RR*0.05, nb)
        for pos in r.choice(nb-1, max(1, int(nb*0.05)), replace=False):
            rv[pos] = r.uniform(0.68, 0.82)*NORMAL_RR
            rv[pos+1] = r.uniform(1.15, 1.35)*NORMAL_RR
        return rv
    return _make_ms(beat, rr, n, 0.04, 0.03, rng)

def synth_afib(n, rng):
    def beat(t, r):
        bl = sum(r.uniform(0.03,0.08)*np.sin(2*np.pi*ff*t+r.uniform(0,6.28))
                 for ff in r.uniform(4,10,6))
        o = r.uniform(-0.05, 0.05)
        return (_gauss(t,o-0.04,0.025,-0.12)+_gauss(t,o,0.035,0.90)
                +_gauss(t,o+0.04,0.025,-0.08)+_gauss(t,o+0.30,0.09,0.25)+bl)
    def rr(nb, r): return r.uniform(0.50*NORMAL_RR, 1.50*NORMAL_RR, nb)
    return _make_ms(beat, rr, n, 0.06, 0.05, rng)

def synth_vt(n, rng):
    def beat(t, r):
        w = r.uniform(0.10, 0.16); p = r.choice([-1, 1])
        return (_gauss(t,0.00,w,p*0.95)+_gauss(t,0.08,w*0.9,p*0.30)
                +_gauss(t,0.35,0.14,-p*0.35))
    def rr(nb, r): return r.normal(144, 4, nb)
    return _make_ms(beat, rr, n, 0.04, 0.03, rng)

def synth_svt(n, rng):
    def beat(t, r): return (_gauss(t,-0.02,0.030,-0.10)+_gauss(t, 0.00,0.035, 1.00)
                            +_gauss(t, 0.04,0.025,-0.08)+_gauss(t, 0.22,0.070, 0.25)
                            +_gauss(t, 0.18,0.040,-0.10))
    def rr(nb, r): return r.normal(120, 3, nb)
    return _make_ms(beat, rr, n, 0.03, 0.02, rng)

SYNTH_FNS = [synth_normal, synth_pvc, synth_afib, synth_vt, synth_svt]

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(n_per_class=300):
    print("\n=== Loading MIT-BIH records (multi-scale features) ===")
    real = load_mitbih_beats(max_per_class=n_per_class)
    rng  = np.random.default_rng(42)
    X, y = [], []
    for cls in range(NUM_CLASSES):
        rb = real.get(cls, []); n_real = len(rb)
        n_syn = max(0, n_per_class - n_real)
        for b in rb:          X.append(b); y.append(cls)
        for b in SYNTH_FNS[cls](n_syn, rng): X.append(b); y.append(cls)
        print(f"  Class {cls} ({CLASS_NAMES[cls]:7s}): "
              f"{n_real:3d} real + {n_syn:3d} synthetic")
    return np.array(X, np.float32), np.array(y, np.int32)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  OvO reconstruction helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_ovo_fns(clf):
    n_cls        = len(clf.classes_)
    sv_class_idx = np.concatenate([np.full(nc, i)
                                   for i, nc in enumerate(clf.n_support_)])

    def reconstruct(K_mat):
        n_test  = K_mat.shape[0]
        n_pairs = n_cls * (n_cls - 1) // 2
        dec = np.zeros((n_test, n_pairs))
        p = 0
        for a in range(n_cls):
            for b in range(a + 1, n_cls):
                sa = sv_class_idx == a; sb = sv_class_idx == b
                dec[:, p] = (K_mat[:, sa] @ clf.dual_coef_[b-1, sa]
                           + K_mat[:, sb] @ clf.dual_coef_[a,   sb]
                           + clf.intercept_[p])
                p += 1
        return dec

    def vote(dec):
        votes = np.zeros((dec.shape[0], n_cls), dtype=int)
        p = 0
        for a in range(n_cls):
            for b in range(a + 1, n_cls):
                mask = dec[:, p] > 0
                votes[ mask, a] += 1
                votes[~mask, b] += 1
                p += 1
        return clf.classes_[np.argmax(votes, axis=1)]

    return reconstruct, vote

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_cm(ax, cm, title, fig):
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted", ylabel="True")
    ax.set_title(title, fontsize=10)
    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > thresh else "black")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"EXP_INT_LUT = {EXP_INT_LUT}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    X, y = build_dataset(n_per_class=300)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # ── sklearn SVM ──────────────────────────────────────────────────────────
    gamma = DEFAULT_GAMMA
    print(f"\n=== Training sklearn RBF SVM  gamma={gamma}  C=1.0 ===")
    clf = SVC(kernel="rbf", gamma=gamma, C=1.0,
              decision_function_shape="ovr", random_state=42)
    clf.fit(X_tr, y_tr)
    print(f"  Total SVs: {clf.n_support_.sum()}  per class: "
          + " ".join(f"{CLASS_NAMES[i]}={clf.n_support_[i]}"
                     for i in range(NUM_CLASSES)))

    y_pred_sk = clf.predict(X_te)
    sk_acc    = accuracy_score(y_te, y_pred_sk)
    print(f"  sklearn accuracy: {sk_acc:.4f}")

    # ── HW kernel matrix (Numba) ─────────────────────────────────────────────
    sv_all  = clf.support_vectors_          # all sklearn SVs
    n_sv    = len(sv_all)
    N_eval  = len(X_te)

    # Q6.10 arrays
    gamma_q  = int(float_to_q10(gamma))
    X_q_te   = vecs_to_q10(X_te)           # (N_eval, 256) int32
    SV_q     = vecs_to_q10(sv_all)          # (n_sv,   256) int32

    # JIT warm-up (compiles on small arrays, cached for next run)
    print(f"\n=== HW kernel matrix: {N_eval} × {n_sv} SVs (Numba JIT) ===")
    _d = np.zeros((1, 256), dtype=np.int32)
    print("  Compiling Numba kernels (first run only)...", flush=True)
    _ = compute_kernel_matrix_nb(_d, _d, gamma_q, _EXP_LUT_NB, _HORNER_NB)
    print("  JIT ready.", flush=True)

    t0    = time.perf_counter()
    K_hw  = compute_kernel_matrix_nb(X_q_te, SV_q, gamma_q,
                                     _EXP_LUT_NB, _HORNER_NB)
    elapsed = time.perf_counter() - t0
    print(f"  Computed in {elapsed:.2f}s  (shape {K_hw.shape})")

    # ── OvO reconstruction ───────────────────────────────────────────────────
    reconstruct, vote = make_ovo_fns(clf)
    dec_hw        = reconstruct(K_hw)
    y_pred_hw     = vote(dec_hw)
    hw_acc        = accuracy_score(y_te, y_pred_hw)
    print(f"  HW accuracy (Q6.10 LUT drain-fixed + OvO): {hw_acc:.4f}")

    # ── Confusion matrices ───────────────────────────────────────────────────
    cm_sk = confusion_matrix(y_te, y_pred_sk, labels=list(range(NUM_CLASSES)))
    cm_hw = confusion_matrix(y_te, y_pred_hw, labels=list(range(NUM_CLASSES)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    plot_cm(axes[0], cm_sk, fig=fig,
            title=f"sklearn  float64\nAccuracy {sk_acc:.2%} — {N_eval} samples")
    plot_cm(axes[1], cm_hw, fig=fig,
            title=f"HW Q6.10  LUT range-reduction  (drain-fixed RTL)\nAccuracy {hw_acc:.2%}")

    fig.suptitle(
        "Multi-scale SVM — Hardware vs Software  (γ=0.25, LUT range-reduction, "
        "256-dim drain fix)\n"
        "ECE 410  ·  5-class cardiac arrhythmia  ·  MIT-BIH  ·  "
        "128 beat + 64 mean + 64 RR features",
        fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "confusion_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {out_path}")

    # ── Per-class summary ────────────────────────────────────────────────────
    print(f"\n  {'Class':<8}  sklearn float64        HW Q6.10 drain-fixed")
    print(  "  " + "-"*55)
    for c, name in enumerate(CLASS_NAMES):
        sup   = cm_sk[c].sum()
        tp_sk = cm_sk[c, c]; tp_hw = cm_hw[c, c]
        print(f"  {name:<8}  {tp_sk:3d}/{sup:3d} ({tp_sk/sup:5.1%})"
              f"        {tp_hw:3d}/{sup:3d} ({tp_hw/sup:5.1%})")
    print(f"\n  Overall  sklearn={sk_acc:.4f}   HW={hw_acc:.4f}"
          f"   gap={abs(sk_acc-hw_acc):.4f}")


if __name__ == "__main__":
    main()
