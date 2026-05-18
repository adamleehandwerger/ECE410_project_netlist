# SVM Compute Core — Error Code Reference

**RTL:** `svm_compute_core.sv`  
**Verification status:** 13/13 unit testbenches PASS  
**Milestone:** m3 (pre-netlist verified)

Error codes are reported on the 4-bit `error_code[3:0]` output and the 1-bit `error` flag.  
Two categories exist: **sticky faults** (0x1–0x7) and **advisory codes** (0x8–0xB).

---

## Quick Reference

| Code | Name | Category | `error` flag | Clears on | Blocks `start`? | Aborts run? |
|------|------|----------|-------------|-----------|-----------------|-------------|
| `0x0` | `ERR_NONE` | — | 0 | — | No | No |
| `0x1` | `ERR_SV_ZERO` | Sticky | 1 | `rst_n` | Yes (never starts) | Yes |
| `0x2` | `ERR_SV_OVERFLOW` | Sticky | 1 | `rst_n` | Yes (never starts) | Yes |
| `0x3` | `ERR_ILLEGAL_STATE` | Sticky | 1 | `rst_n` | Yes | Yes |
| `0x4` | `ERR_GAMMA_SAT` | Sticky | 1 | `rst_n` | No | No |
| `0x5` | `ERR_FIFO_OVERFLOW` | Sticky | 1 | `rst_n` | No | No |
| `0x6` | `ERR_GAMMA_ZERO` | Sticky | 1 | `rst_n` | No | No |
| `0x7` | `ERR_NUM_SAMPLES_ZERO` | Sticky | 1 | `rst_n` | Yes (never starts) | Yes |
| `0x8` | `ERR_WARMING_UP` | Advisory | 1 | Beat 100 / sticky fault | No | No |
| `0x9` | `ERR_INTERRUPTED` | Advisory | 1 | Beat 100 / sticky fault | No | No |
| `0xA` | `ERR_LOW_BATTERY` | Advisory | 1 | `vbatt_warn` deasserts | No | No |
| `0xB` | `ERR_POWER_FAIL` | Advisory | 1 | `vbatt_ok` reasserts | Yes (new starts only) | No |

**Advisory check in RTL/host code:** `error_code >= 4'h8` → advisory (not a hard fault).

---

## Category Rules

### Sticky Faults (0x1–0x7)

- `error` latches to 1 and **never self-clears**.
- `error_code` holds the first fault code that fired.
- Only `rst_n` (active-low reset pulse) clears both.
- A batch that was running when the fault fires is considered invalid; kernel outputs should be discarded.
- Sticky faults always win over advisory codes — if both are present simultaneously, the sticky code is reported.

### Advisory Codes (0x8–0xB)

- `error` asserts while the condition is active, clears automatically when it resolves.
- `error_code` reverts to `ERR_NONE` (0x0) when cleared, unless a sticky fault is also latched.
- A running batch is **not aborted** (except `ERR_POWER_FAIL` which blocks new starts only).
- If a sticky fault fires while an advisory is active, the sticky code immediately overrides and latches.

---

## Sticky Faults — Detail

### 0x1 — ERR_SV_ZERO

**Trigger:** `Σ num_sv_per_class == 0` when `start` is asserted.

No support vectors are loaded. The FSM would immediately terminate with zero kernel outputs, which is an undefined classifier state. The batch is rejected and the FSM stays in `IDLE`.

**Host response:** Reload `num_sv_per_class` with valid per-class counts (each ≥ 0, sum ≥ 1) before asserting `start` again. Assert `rst_n` to clear the fault.

---

### 0x2 — ERR_SV_OVERFLOW

**Trigger:** `Σ num_sv_per_class > NUM_SV` (default: 250) when `start` is asserted.

The address space of `sv_ram` is sized for `NUM_SV` vectors. An overflow would read past the end of the SV SRAM. The batch is rejected.

**Host response:** Reduce per-class SV counts so the total does not exceed `NUM_SV`. Assert `rst_n` to clear.

---

### 0x3 — ERR_ILLEGAL_STATE

**Trigger:** FSM `default` branch taken — the state register holds an unrecognised value.

This is an internal fault (bit-flip, glitch, synthesis issue). The FSM transitions to `ERROR_STATE` for one cycle, latches the code, then returns to `IDLE`.

**Host response:** Assert `rst_n`. If the fault recurs, the RTL or netlist has a structural problem.

---

### 0x4 — ERR_GAMMA_SAT

**Trigger:** `gamma_int > 8192 (0x2000)` while the FSM is **not** in `IDLE`.

At γ > 8.0 the product `γ × dist` overflows the 20-bit distance accumulator at full scale. Kernel outputs are meaningless. The batch continues to completion (the `gamma_latched` shadow register was already captured at `start` with the last valid value), but results should be discarded.

> **Note:** The write to `gamma_reg` is not blocked. `ERR_GAMMA_SAT` fires on the *next* compute cycle that sees the illegal value, not at write time. Writing a saturating gamma while the FSM is in `IDLE` is safe — the shadow register captures `gamma_int` at `start`, so the fault only fires if the FSM is already running.

**Host response:** Write a valid γ (≤ 8.0 in float, ≤ `0x2000` in Q6.10). Assert `rst_n` to clear the sticky flag before restarting.

---

### 0x5 — ERR_FIFO_OVERFLOW

**Trigger:** `qspi_valid` asserted when the input FIFO is full (`fifo_count == FIFO_DEPTH`). The arriving word is dropped.

`qspi_ready` deasserts when the FIFO is full — the external deserializer should observe backpressure. If it does not (e.g., an SPI-to-parallel bridge without ready-signal support), words are silently lost and feature vectors become corrupted.

**Host response:** Respect `qspi_ready`. After `rst_n`, verify that the QSPI interface honours backpressure before re-streaming feature data.

---

### 0x6 — ERR_GAMMA_ZERO

**Trigger:** `gamma_int == 0` while the FSM is **not** in `IDLE`.

γ = 0 collapses the RBF kernel to exp(0) = 1.0 for every support vector regardless of distance, making the classifier output a constant. Unlike `ERR_GAMMA_SAT`, the computation completes without numerical overflow — the error is semantic, not arithmetic. The batch still produces kernel outputs, all equal to 1024 (1.0 in Q6.10).

**Host response:** Write a non-zero γ. Assert `rst_n` to clear. The completed batch's kernel outputs should be discarded (constant classifier).

---

### 0x7 — ERR_NUM_SAMPLES_ZERO

**Trigger:** `num_samples == 0` when `start` is asserted.

`last_heartbeat` is computed as `sample_counter >= num_samples_latched - 1`. With `num_samples_latched = 0`, the subtraction underflows to `1023` (10-bit unsigned), so `last_heartbeat` is never true and the batch loops forever. The FSM is preemptively blocked.

**Host response:** Set `num_samples ≥ 1` before asserting `start`. Assert `rst_n` to clear.

---

## Advisory Codes — Detail

### 0x8 — ERR_WARMING_UP

**Trigger:** `heartbeat_count == 0` at the time of a clean power-on reset (`rst_n` deasserted with no prior incomplete warm-up).

The 256-dim feature vector has three slices:

| Slice | Dims | Window |
|-------|------|--------|
| Single-beat morphology | 128 | ±64 samples around R-peak |
| 10-beat mean morphology | 64 | mean of ±32-sample snippets, 10 surrounding beats |
| 100-beat RR-interval track | 64 | 99 RR intervals centred on target beat |

The 10-beat and 100-beat slices require history that only accumulates after enough heartbeats have been processed. Results from the first 99 beats are unreliable.

**Auto-clears:** when `heartbeat_count` reaches 100.  
**Overridden by:** any sticky fault (0x1–0x7).

**Host response:** Flag or discard classification results while `ERR_WARMING_UP` is active. Once the code clears (beat 100+), results are fully reliable.

---

### 0x9 — ERR_INTERRUPTED

**Trigger:** `rst_n` is pulsed while `heartbeat_count` is in `[1, 99]` — i.e., a warm-up was in progress but not complete.

Without this code, the host cannot distinguish a fresh power-on (`heartbeat_count == 0` → `ERR_WARMING_UP`) from a disrupted session. `ERR_INTERRUPTED` fires *instead of* `ERR_WARMING_UP` in the interrupted case.

**Auto-clears:** when `heartbeat_count` reaches 100 (same as `ERR_WARMING_UP`).  
**Overridden by:** any sticky fault (0x1–0x7).

**Host response:** Same as `ERR_WARMING_UP` — discard results until beat 100. Optionally log the interruption for diagnostics.

---

### 0xA — ERR_LOW_BATTERY

**Trigger:** `vbatt_warn_s == 1` (synchronized `vbatt_warn` input).

`vbatt_warn` is driven by an analog comparator connected to the battery voltage rail. Assertion indicates the battery is below the soft warning threshold but still above the hard operational minimum. The device continues to function normally.

**Auto-clears:** when `vbatt_warn` deasserts (2-cycle synchronizer delay).  
**Overridden by:** any sticky fault (0x1–0x7).

**Host response:** Notify the user or upstream system of low battery. Complete the current classification session if possible. Do not start a new multi-hour batch.

---

### 0xB — ERR_POWER_FAIL

**Trigger:** `vbatt_ok_s == 0` (synchronized `vbatt_ok` input).

`vbatt_ok` is driven by an analog comparator at the hard operational threshold. Deassertion means the supply voltage is insufficient to guarantee correct digital operation. The FSM blocks new `start` pulses but does **not** abort a run already in progress — an in-flight classification is preferable to a half-finished one.

**Auto-clears:** when `vbatt_ok` reasserts (2-cycle synchronizer delay).  
**Overridden by:** any sticky fault (0x1–0x7).

> **Synchronizer reset value:** `vbatt_ok` synchronizer resets to `1` at POR so the FSM does not see a spurious `ERR_POWER_FAIL` during the reset cycle.

**Host response:** Do not issue new `start` pulses while `ERR_POWER_FAIL` is active. Wait for `vbatt_ok` to reassert (charge cycle, power-supply restore). The current batch will complete; its results are valid if no sticky fault is also raised.

---

## Priority and Override Rules

```
Highest priority
      │
      ▼   ERR_SV_ZERO          (0x1)  ─┐
          ERR_SV_OVERFLOW       (0x2)   │  Sticky: latch on first
          ERR_NUM_SAMPLES_ZERO  (0x7)   │  occurrence; hold until
          ERR_ILLEGAL_STATE     (0x3)   │  rst_n; override all
          ERR_GAMMA_SAT         (0x4)   │  advisory codes
          ERR_GAMMA_ZERO        (0x6)   │
          ERR_FIFO_OVERFLOW     (0x5)  ─┘
          ─────────────────────────────
          ERR_POWER_FAIL        (0xB)  ─┐
          ERR_LOW_BATTERY       (0xA)   │  Advisory: auto-clear;
          ERR_INTERRUPTED       (0x9)   │  pre-empted by any
          ERR_WARMING_UP        (0x8)  ─┘  sticky fault above
      │
      ▼
Lowest priority
```

When multiple conditions are active simultaneously:
- The highest-priority **sticky** fault wins and latches.
- Among advisories, `ERR_POWER_FAIL` (0xB) takes precedence over the others.
- Once a sticky fault latches, advisory codes are suppressed until after `rst_n`.

---

## Example Sequences

### Cold start, normal operation

```
rst_n deasserts          → error_code = 0x8 (ERR_WARMING_UP)
beats 1–99 processed     → error_code = 0x8 (advisory, results flagged)
beat 100 processed       → error_code = 0x0 (ERR_NONE, warm-up complete)
beats 101+               → normal classification, no error
```

### Reset during warm-up

```
rst_n deasserts (beat 0) → error_code = 0x8 (ERR_WARMING_UP)
beats 1–47 processed     → error_code = 0x8
rst_n pulsed at beat 47  → arm_interrupted latched
rst_n deasserts again    → error_code = 0x9 (ERR_INTERRUPTED)
beats 1–99 processed     → error_code = 0x9 (advisory)
beat 100 processed       → error_code = 0x0 (cleared)
```

### Low battery during run

```
vbatt_warn asserts       → error_code = 0xA (ERR_LOW_BATTERY)
start + run proceeds     → FSM runs normally; kernel outputs valid
done fires               → batch complete
vbatt_warn deasserts     → error_code = 0x0 (auto-cleared)
```

### Power fail blocks start

```
vbatt_ok deasserts       → error_code = 0xB (ERR_POWER_FAIL)
host asserts start       → FSM stays in IDLE; start ignored
vbatt_ok reasserts       → error_code = 0x0 (auto-cleared)
host asserts start       → FSM proceeds normally
```

### Sticky fault overrides advisory

```
vbatt_warn asserts       → error_code = 0xA (ERR_LOW_BATTERY)
gamma written to 9000    → ERR_GAMMA_SAT fires mid-compute
                         → error_code = 0x4 (sticky, overrides 0xA)
vbatt_warn deasserts     → error_code stays 0x4 (sticky holds)
rst_n pulsed             → error_code = 0x0 (cleared)
```

---

## Testbench Coverage

| Testbench | Codes exercised |
|-----------|----------------|
| `tb_error_codes.sv` | 0x1, 0x2, 0x4, 0x5 — sticky latch; reset clears |
| `tb_gamma_zero.sv` | 0x6 |
| `tb_warmup.sv` | 0x8, 0x9 — cold start; interrupted; sticky override; beat-100 auto-clear |
| `tb_power.sv` | 0xA, 0xB — advisory behaviour; start blocking; FSM completion; sticky override |
| `tb_param_write.sv` | 0x4 — mid-pipeline gamma write |
| `tb_backpressure.sv` | 0x0 — verifies no spurious errors during backpressure |
| `tb_svm_classifier.sv` | 0x0 — full 5-class run; no error flag |
