# SVM Compute Core — Design Summary

**Project:** Multi-Class Cardiac Arrhythmia Detection  
**RTL:** `svm_compute_core.sv` (m3, pre-netlist)  
**Accuracy:** 98.67% on MIT-BIH (sklearn = HW, zero gap)

Each section below describes one design component, why the chosen configuration is optimal for this application, and how two alternatives compare.

---

## 1. Fixed-Point Format — Q6.10 (16-bit)

**Chosen:** 16-bit signed fixed-point, 10 fractional bits (Q6.10). Integer range ±32, LSB ≈ 0.001. γ = 0.25 maps exactly to raw 256 = `0x0100`.

**Why optimal:** A 16-bit datapath fits a single DSP slice or multiplier primitive on any target technology. The 10-bit fraction gives ~3 decimal digits of precision — sufficient for SVM kernel values in [0, 1] where classification boundaries are stable. γ = 0.25 is exactly representable, eliminating quantization error in the most sensitivity-affecting parameter.

| Alternative | Trade-off |
|-------------|-----------|
| **Q8.8** (8 fractional bits) | 2× coarser kernel resolution; γ = 0.25 still exact, but small distance differences (~0.004) collapse to the same bin. Measurable accuracy drop at decision boundaries between morphologically similar classes (PVC vs. Normal). |
| **Float32** | Exact arithmetic, no quantization. Requires a full FP multiply-add unit: ~4–8 DSPs on FPGA, ~10× gate count in ASIC, >10× power. Unjustifiable for a wearable running on a coin cell. |

---

## 2. Feature Vector — 256-dim Multi-Scale

**Chosen:** Three concatenated slices totalling 256 dimensions:
- 128-dim single-beat morphology (±64 samples around R-peak)
- 64-dim 10-beat mean morphology (mean of ±32-sample snippets, 10 beats)
- 64-dim 100-beat RR-interval track (99 intervals)

**Why optimal:** Cardiac arrhythmia classification requires both instantaneous morphology (discriminates PVC, SVT from Normal in a single beat) and rhythm context (AFib requires irregular RR over many beats; VT requires sustained tachycardia). The three-slice design provides all three timescales with no redundancy. 256 is a power of two, which eliminates modular addressing overhead and makes the accumulator loop a clean 256-cycle count.

| Alternative | Trade-off |
|-------------|-----------|
| **Single-beat morphology only (128-dim)** | Reduces FIFO depth requirement by half and cuts compute by 2×. Accurately identifies PVC and SVT but cannot reliably distinguish AFib (needs RR irregularity) or VT (needs sustained run context). Expected accuracy drop ~4–6% on MIT-BIH for those two classes. |
| **Continuous wavelet features (256-dim)** | Better time-frequency localization, proven in literature for arrhythmia. Requires a CWT preprocessing block (~3× more compute, multi-stage filter bank) that is not feasible on a low-power wearable without a dedicated DSP. Sample-domain features computed directly from the ADC stream are simpler and lower power. |

---

## 3. Kernel Function — RBF with γ = 0.25

**Chosen:** Radial basis function (Gaussian) kernel `K(x, sv) = exp(-γ ||x - sv||²)`, γ = 0.25.

**Why optimal:** RBF is the standard kernel for non-linearly separable biosignal classification. γ = 0.25 was selected by cross-validation on MIT-BIH; it gives a kernel width of ~2 in feature space, providing enough localisation to distinguish PVC morphology from Normal without overfitting. At this γ, the exponent argument `γ·D` stays below 16 for almost all physically realizable feature pairs, which the range-reduction LUT exploits exactly.

| Alternative | Trade-off |
|-------------|-----------|
| **Linear kernel** | No kernel computation needed — classification reduces to a dot product. Training accuracy on MIT-BIH is ~91%, roughly 7 points below RBF, because arrhythmia class boundaries are non-linear in the feature space. |
| **Polynomial kernel (degree 3)** | Competitive accuracy for some ECG datasets. Requires computing `(x·sv + r)³`, which involves two full 256-dim dot products plus a cube operation. Gate count and latency are similar to RBF, but the polynomial is more sensitive to feature scaling and outlier SVs. RBF generalises better with fewer support vectors. |

---

## 4. Kernel Approximation — Range-Reduction LUT + 15th-Order Horner

**Chosen:** Factored evaluation `exp(-γD) = exp(-I) × exp(-F)` where I is the integer part and F is the fractional part of the scaled distance. `exp(-I)` is a 16-entry read-only LUT; `exp(-F)` is a 15th-order Horner polynomial over `F ∈ [0, 1)`.

**Why optimal:** A naive single-stage Horner over the full range of `γD` (which can reach ~16) overflows int16 at the high end — the polynomial wraps and incorrectly returns 1.0, producing a near-random classifier (~20% accuracy). Range-reduction confines the Horner input to [0, 1), where all 15 coefficients fit in Q6.10 and the polynomial is numerically stable. The LUT adds only 16 × 16-bit = 32 bytes of ROM and zero latency cycles (combinational lookup in SCALE2). Total Horner latency is 18 cycles — identical to the old single-stage version.

| Alternative | Trade-off |
|-------------|-----------|
| **CORDIC exp** | Hardware-friendly iterative algorithm; no LUT needed. Requires ~16 iterations for 16-bit precision, each with an adder and shift — roughly 16 cycles and more control logic than Horner. CORDIC is better suited for trigonometric functions; for exp it does not converge as cleanly and requires pre-scaling anyway. |
| **256-entry full LUT (direct lookup on quantized γD)** | Zero arithmetic — just index the table. With 20-bit distance and Q6.10 gamma, the product has 26 effective bits; a direct table would need 64 K entries = 128 KB of ROM, larger than the SV RAM itself. Impractical on any wearable target. |

---

## 5. Distance Matrix Engine — 20-bit Accumulator, 2-Stage Pipeline

**Chosen:** Sequential accumulator over 256 dimensions, 20-bit saturating output. Pipeline: `diff → diff_squared → accumulator` (2-cycle latency). Two explicit drain cycles after the last valid_in flush all 256 contributions.

**Why optimal:** 20 bits covers the worst-case squared Euclidean distance for 256-dim Q6.10 vectors without saturation under normal conditions (max theoretical ~256 × 32² / 1024 = 262,144 < 2²⁰). A purely combinational 256-input adder tree would need 512 adders and 8 pipeline stages just for the reduction, with a 256-cycle-wide datapath — area and routing impractical. The sequential accumulator uses one multiplier and one adder, reused 256+2 times. The drain flush (2 extra cycles) was critical: without it, the last 2 dimensions were silently dropped every kernel computation.

| Alternative | Trade-off |
|-------------|-----------|
| **16-bit accumulator** | Saves two flip-flops per bit across the accumulator width. However, 16 bits saturates at 65,535 ≈ 64 features × 32² / 1024, meaning any feature vector with large components in the last 192 dimensions would clip. Distance comparisons become unreliable for distant class pairs. |
| **Parallel 4-way unrolled (64-cycle compute)** | 4× throughput — reduces compute time from 258 to ~66 cycles. Requires 4 multipliers, 4 adders, and 4 SV RAM read ports operating simultaneously. Area roughly 4× the current engine. For a duty-cycled wearable running one classification per heartbeat (~1 Hz), the extra throughput has no value and wastes leakage power. |

---

## 6. Input FIFO — 8192-word (16 KB) On-Chip SRAM

**Chosen:** Synchronous single-clock FIFO, depth 8192 × 16-bit words (16 KB), with backpressure via `qspi_ready`.

**Why optimal:** QSPI streams data at 1 M words/sec (256 µs per heartbeat). The core processes one SV against 256 features in 258 cycles at the system clock (assume 1–4 MHz). With 250 SVs × 5 classes, compute takes 250 × 258 = 64,500 cycles. At 4 MHz that is 16 ms — 62× longer than the 256 µs data window. The FIFO absorbs the full burst and allows the core to process at its own rate without dropping samples. Depth 8192 = 32 heartbeats of margin, sufficient for multi-heartbeat batch operation. Backpressure (`qspi_ready`) prevents silent data loss and triggers `ERR_FIFO_OVERFLOW` if the host does not respect it.

| Alternative | Trade-off |
|-------------|-----------|
| **Double buffer (2 × 256 words)** | Minimal area — only 1 KB. Requires the host MCU to alternate between two 256-word slots in strict lockstep with the core. Any compute stall (gamma update, error handling) causes a dropped heartbeat. Fragile for a medical device. |
| **External FIFO (off-chip)** | Unlimited depth. Adds 1–2 cycles of off-chip latency per word read, turning the `LOAD_FEATURES` state into a memory-latency-bound stage. More pins, more power, defeats the low-cost wearable target. |

---

## 7. Feature Register Bank — On-Chip Register Array (256 × 16-bit)

**Chosen:** 256-word on-chip register array (`feature_bank`), loaded from FIFO during `LOAD_FEATURES`, read sequentially during `COMPUTE_DIST`.

**Why optimal:** The same 256-dim feature vector is read once for every support vector (up to 250 times per heartbeat). Reading from off-chip SRAM 250 × 256 = 64,000 times per heartbeat at 16-bit width would saturate an off-chip bus and create a bottleneck larger than the compute itself. Keeping features on-chip as a register array gives single-cycle access with no bus arbitration. At 256 × 16 = 512 bytes it is small enough to infer as flip-flop array or local SRAM on any target.

| Alternative | Trade-off |
|-------------|-----------|
| **Re-read from FIFO each SV** | Eliminates the register bank (saves 512 B). But once 256 words are consumed from the FIFO, they are gone — requires re-streaming from the MCU for each SV, multiplying QSPI traffic by 250×. Not feasible. |
| **Store features in Workspace RAM (off-chip)** | Saves on-chip area. Workspace RAM has 1-cycle synchronous latency but still requires a dedicated read bus and conflicts with kernel output writes in `OUTPUT_RESULT`. The additional bus-arbitration logic costs more area than the 256-register bank it replaces. |

---

## 8. SV RAM — Off-Chip, 128 KB, Read-Only

**Chosen:** Off-chip synchronous SRAM, 250 × 256 × 2 B = 128 KB, 18-bit address, read-only from the core. Loaded once at device initialisation.

**Why optimal:** 128 KB exceeds any reasonable on-chip SRAM budget for a wearable ASIC or small FPGA. SVs are fixed after training and never written during inference, so a read-only interface (addr + ren + rdata) minimises pin count and avoids write-enable routing. The 18-bit flat address space `{sv_index[9:0], feature_index[7:0]}` maps directly to the sequential read pattern used by `COMPUTE_DIST`, eliminating any address translation logic.

| Alternative | Trade-off |
|-------------|-----------|
| **Compressed SVs (8-bit quantized, 64 KB)** | Halves the memory footprint. Q8 SVs would require the distance engine to sign-extend or convert on the fly, adding a cycle and logic. More importantly, 8-bit SVs introduce quantization error that widens the RBF kernel's effective γ and reduces decision-boundary sharpness — measured ~1–2% accuracy loss on MIT-BIH at Q6.10 features. |
| **On-chip ROM (embedded flash or SRAM)** | Eliminates off-chip package pins and reduces power for the SV reads. Feasible on larger ASICs (180 nm and above typically support 128 KB+ embedded SRAM). Adds significant die area and makes re-training (different patient model) require re-programming the device. Off-chip flash enables field-updateable SVM models. |

---

## 9. Workspace RAM — Off-Chip, ≤500 KB, Read/Write

**Chosen:** Off-chip SRAM, 19-bit address, read/write. Stores kernel outputs as they are produced: one 16-bit word per (sample, SV) pair, up to 1000 × 250 × 2 B = 500 KB.

**Why optimal:** The host MCU reads kernel outputs after `done` and performs the final decision function (weighted sum + bias) in software. Storing outputs in external RAM decouples the compute core from the decision logic entirely — the core never needs to hold all outputs simultaneously. The 19-bit address covers the full 500 KB range; sequential kernel output write order (incrementing `kernel_out_counter`) matches the address pattern expected by the MCU.

| Alternative | Trade-off |
|-------------|-----------|
| **Stream directly to MCU, no RAM** | Eliminates workspace RAM. Requires the MCU to receive and accumulate 250 kernel values per SV per heartbeat in real time, at up to 250 × 258 cycles per heartbeat. This forces the MCU to match the core's clock and removes all buffering — impractical for an MCU that also handles display, BLE, and data logging. |
| **Accumulate decision function on-chip** | Core computes `Σ αᵢ K(x, svᵢ)` internally and outputs only a 5-bit class label. Eliminates the workspace RAM entirely and reduces the host interface to `done + class_out[2:0]`. Downside: the decision coefficients αᵢ (250 per class × 10 OvO pairs = 2500 values × 2 B = 5 KB) must be loaded alongside the SVs; the accumulator logic for 5 simultaneous class scores adds area; and the host loses access to raw kernel values for confidence scoring and diagnostics. |

---

## 10. FSM Architecture — 7-State Sequential

**Chosen:** Single top-level FSM with 7 states: IDLE, LOAD_FIFO, LOAD_FEATURES, COMPUTE_DIST, COMPUTE_KERNEL, OUTPUT_RESULT, ERROR_STATE. Submodules (`distance_matrix`, `horner_engine`) have their own internal FSMs.

**Why optimal:** The workload is strictly sequential within a heartbeat — there is no opportunity for top-level parallelism between FIFO loading, feature loading, distance computation, and kernel computation. A single FSM makes control flow auditable, directly testable state by state, and synthesises to a small one-hot or binary encoder. Offloading ACCUMULATE and HORNER states to submodule FSMs keeps the top-level transition logic under 10 cases and avoids a monolithic 25+ state machine that would be error-prone to modify.

| Alternative | Trade-off |
|-------------|-----------|
| **Pipelined (overlap LOAD_FIFO for next SV while computing current)** | Potential 2× throughput for the FIFO-loading stage. Requires a double-buffered feature bank and arbitration between write (next heartbeat) and read (current SV). Adds ~200 lines of control logic. For 1 Hz wearable operation the throughput gain has no user-visible benefit and the added complexity increases verification burden significantly. |
| **Microcode / sequencer ROM** | Replace the FSM with a small instruction ROM and program counter. Highly flexible for adding new states without RTL changes. Adds a fetch-decode path, increases cycle count by ~2× (each "instruction" costs a cycle), and makes formal verification harder. Appropriate for a general-purpose compute engine, not for a fixed SVM pipeline. |

---

## 11. Error Handling — Sticky Faults + Advisory Codes

**Chosen:** Two-tier error system. Sticky faults (0x1–0x7) latch on first occurrence, hold until `rst_n`, always override advisory codes. Advisory codes (0x8–0xB) auto-clear when the triggering condition resolves.

**Why optimal:** Medical-grade firmware must distinguish "computation is invalid, discard results" (sticky) from "results are valid but context is degraded" (advisory). Conflating them into a single sticky register would force a full reset on a low-battery warning, disrupting ongoing classification. Separating the two tiers lets the host firmware make a graded response: reset and reinitialise on a sticky fault; log and warn on an advisory. The 4-bit `error_code` encodes both categories in a single output; the advisory boundary at 0x8 (`error_code >= 4'h8`) is a single comparator.

| Alternative | Trade-off |
|-------------|-----------|
| **Single sticky register, no advisory tier** | Simpler — one always_ff block, one comparator in the host. `ERR_WARMING_UP`, `ERR_LOW_BATTERY` and `ERR_POWER_FAIL` would all require an `rst_n` to clear. For a wearable worn 24/7, spurious resets during warm-up (beats 1–99) or low-battery charge cycles would make the device appear to malfunction from the user's perspective. |
| **Separate `advisory_code[3:0]` output, no unified `error_code`** | Clean separation; host reads two registers. Doubles the status output width and requires the host to poll two signals. The current scheme encodes both categories in one 4-bit output with a threshold check, which is simpler for an MCU interrupt handler and reduces the interface pin count by 4. |

---

## 12. Input Synchronizers — 2-FF Barrier for `vbatt_ok` / `vbatt_warn`

**Chosen:** Two-stage flip-flop synchronizer (`sync_ff`, parameterised STAGES=2) for each async comparator input. Reset values: `vbatt_ok → 1` (assume power OK at POR), `vbatt_warn → 0` (no warning at POR).

**Why optimal:** Analog comparator outputs switch asynchronously with respect to the system clock. Driving them directly into a synchronous FSM violates setup/hold time, causing metastability — a flip-flop output can remain in an undefined state for an unbounded time, potentially causing simultaneous 0 and 1 observations downstream. The 2-FF synchronizer reduces the metastability propagation probability to ~10⁻¹⁵ per cycle at typical process corners (well within the wearable's lifetime). The asymmetric reset values prevent a spurious `ERR_POWER_FAIL` at POR (which would block the first `start` pulse) and prevent a spurious `ERR_LOW_BATTERY` on cold start.

| Alternative | Trade-off |
|-------------|-----------|
| **Direct connection (no synchronizer)** | Zero area cost. Functionally correct in simulation (simulators do not model metastability). Fails in silicon: metastability at the comparator transition causes the FSM to see simultaneous IDLE and LOAD_FIFO transitions, corrupting `sv_count_reg` and producing incorrect kernel outputs. Not acceptable for a device making cardiac rhythm decisions. |
| **3-FF synchronizer (STAGES=3)** | Reduces metastability probability by another 3–4 orders of magnitude (~10⁻¹⁸/cycle). Adds one cycle of latency on the battery signals (3 vs. 2 cycles). The `vbatt` signals change on the timescale of seconds (comparator hysteresis + RC filter on the supply); a 1-cycle latency difference at any clock frequency below 100 MHz is invisible. 2 FF is the standard for this application. |

---

*Document version: m3 · 2026-05-18*
