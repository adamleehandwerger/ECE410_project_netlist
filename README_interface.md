# SVM Compute Core — Interface Reference

**RTL:** `svm_compute_core.sv`  
**Interface definitions:** `svm_interfaces.sv`  
**Testbench:** `tb_interface.sv` → `tb_interface.log`  
**Verification status:** 13/13 unit testbenches PASS · 25/25 interface checks PASS  
**Milestone:** m3 (pre-netlist verified)

---

## Physical Boundaries

```
           ┌─────────────────────────────────────────┐
    MCU ───┤  svm_host_if                            │
           │  (QSPI stream, params, control, status, │
           │   kernel output)                        │
           │                         svm_compute_core│
 SV SRAM ──┤  svm_sv_ram_if                          │
           │  (128 KB, read-only)                    │
           │                                         │
Work SRAM ─┤  svm_work_ram_if                        │
           │  (≤500 KB, read/write)                  │
           └─────────────────────────────────────────┘
```

Three SystemVerilog interfaces span the three physical boundaries. For Icarus Verilog testbenches, use flat port wiring (see [Instantiation](#instantiation)).

---

## svm_host_if — MCU ↔ Core

### QSPI Feature Stream

| Signal | Dir (core) | Width | Description |
|--------|-----------|-------|-------------|
| `qspi_valid` | input | 1 | Deserializer asserts when a 16-bit word is ready |
| `qspi_data` | input | 16 | Q6.10 feature word |
| `qspi_ready` | output | 1 | Core deasserts when input FIFO is full |

**Protocol:** SPI Mode 0 (CPOL=0, CPHA=0), 4 data lanes (Quad SPI), 4 MHz SCK

| Parameter | Value |
|-----------|-------|
| Bit rate | 16 Mbps |
| Word rate | 1 M words/sec |
| Byte rate | 2 MB/s |
| One heartbeat | 256 features × 2 B = 512 B → 256 µs transfer |
| FIFO depth | 8192 words (16 KB) |

The deserializer is external; the core sees only the ready-valid bus. Backpressure is via `qspi_ready`.

**QSPI word framing (4 SCK cycles per 16-bit word, MSB first):**

| SCK cycle | IO[3:0] carries |
|-----------|----------------|
| 1 | bits [15:12] |
| 2 | bits [11:8] |
| 3 | bits [7:4] |
| 4 | bits [3:0] |

> **Important:** `fifo_wr_en` is only active when the FSM is in `LOAD_FIFO`. Feature words fed during any other state are silently discarded. For multi-heartbeat batches, the host must wait for the FSM to return to `LOAD_FIFO` before feeding the next heartbeat's data.

---

### Parameter Programming

| Signal | Dir (core) | Width | Description |
|--------|-----------|-------|-------------|
| `param_write_en` | input | 1 | One-cycle write strobe |
| `param_addr` | input | 3 | Register address (see map below) |
| `param_data` | input | 16 | Write data (Q6.10) |
| `gamma_reg` | output | 16 | Readback of γ register |
| `c_reg` | output | 16 | Readback of C register |
| `bias_reg[5]` | output | 16×5 | Readback of per-class bias registers |

#### Register Map

| Address | Register | Reset default | Q6.10 value | Notes |
|---------|----------|---------------|-------------|-------|
| `3'h0` | `gamma_reg` | `0x0100` | 0.25 | RBF bandwidth γ |
| `3'h1` | `c_reg` | `0x0400` | 1.0 | SVM penalty C |
| `3'h2` | `bias_reg[0]` | `0x0000` | 0.0 | Class 0 (Normal) decision bias |
| `3'h3` | `bias_reg[1]` | `0x0000` | 0.0 | Class 1 (PVC) decision bias |
| `3'h4` | `bias_reg[2]` | `0x0000` | 0.0 | Class 2 (AFib) decision bias |
| `3'h5` | `bias_reg[3]` | `0x0000` | 0.0 | Class 3 (VT) decision bias |
| `3'h6` | `bias_reg[4]` | `0x0000` | 0.0 | Class 4 (SVT) decision bias |
| `3'h7` | *(reserved)* | — | — | Writes ignored |

**Q6.10 encoding:** `raw = round(real_value × 1024)`  
Range: −32.000 to +31.999 · LSB ≈ 0.000977

**Gamma saturation:** Writes to `gamma_reg` are always accepted (no write-time rejection). The error encoder raises `ERR_GAMMA_SAT` if `gamma_int > 8192 (0x2000)` while the FSM is **not** in `IDLE`. The write itself is not blocked.

---

### Power / Battery Monitoring

| Signal | Dir (core) | Width | Description |
|--------|-----------|-------|-------------|
| `vbatt_warn` | input | 1 | Battery below soft threshold (async; internally 2-FF synchronized) |
| `vbatt_ok` | input | 1 | Battery above hard operational threshold (async; internally 2-FF synchronized) |

Both signals are driven by analog comparators and pass through 2-FF synchronizers (`sync_ff`) inside the core before reaching the FSM and error encoder. Reset values: `vbatt_ok→1` (assume power OK at POR), `vbatt_warn→0`.

| Condition | Error raised | FSM behaviour |
|-----------|-------------|----------------|
| `vbatt_warn=1` | `ERR_LOW_BATTERY (0xA)` advisory | FSM continues; run completes normally |
| `vbatt_ok=0` | `ERR_POWER_FAIL (0xB)` advisory | Blocks `start`; running batch is not aborted |
| `vbatt_ok` restored | advisory auto-clears | `start` accepted again |

> **Advisory codes (0x8–0xB) are non-sticky.** They auto-clear when the triggering condition is removed. A sticky fault (0x1–0x7) always overrides an advisory. See Error Code Table below.

---

### Batch Control

| Signal | Dir (core) | Width | Description |
|--------|-----------|-------|-------------|
| `num_sv_per_class[5]` | input | 8×5 | SV count per class; evaluated at `start` |
| `start` | input | 1 | One-cycle pulse; valid in IDLE state only (blocked when `vbatt_ok=0`) |
| `num_samples` | input | 10 | Heartbeats in this batch (1–1000) |
| `done` | output | 1 | One-cycle pulse after the last kernel output |

**`num_samples` is latched** into `num_samples_latched` at `start && vbatt_ok_s` (Fix #7). The FSM evaluates `sample_counter >= num_samples_latched - 1` using the shadow copy throughout the batch. Mid-batch writes to `num_samples` are safe and have no effect.

Valid SV count: `1 ≤ Σ num_sv_per_class ≤ NUM_SV`. Zero or overflow raises an error and the batch does not start.

---

### Status

| Signal | Dir (core) | Width | Description |
|--------|-----------|-------|-------------|
| `error` | output | 1 | Sticky flag; set on any fault; cleared only by `rst_n` |
| `error_code` | output | 4 | Latched fault code (see table below) |

#### Error Code Table

| Code | Name | Trigger |
|------|------|---------|
| `0x0` | `ERR_NONE` | No fault |
| `0x1` | `ERR_SV_ZERO` | Σ `num_sv_per_class` = 0 at `start` |
| `0x2` | `ERR_SV_OVERFLOW` | Σ `num_sv_per_class` > `NUM_SV` at `start` |
| `0x3` | `ERR_ILLEGAL_STATE` | FSM default branch taken (internal fault) |
| `0x4` | `ERR_GAMMA_SAT` | `gamma_int > 8192` while FSM is not `IDLE` |
| `0x5` | `ERR_FIFO_OVERFLOW` | QSPI data arrived when FIFO full (word dropped) |
| `0x6` | `ERR_GAMMA_ZERO` | `gamma_int = 0` while FSM is not `IDLE` (silent classifier failure — all kernels collapse to 1.0) |
| `0x7` | `ERR_NUM_SAMPLES_ZERO` | `num_samples = 0` at `start` — `last_heartbeat` underflows to 1023; batch never terminates |
| `0x8` | `ERR_WARMING_UP` *(advisory)* | Non-sticky; fires on clean cold start (`heartbeat_count` = 0 at POR). Auto-clears at beat 100. Early results may be unreliable (10-beat and 100-beat feature slices incomplete). |
| `0x9` | `ERR_INTERRUPTED` *(advisory)* | Non-sticky; fires instead of `ERR_WARMING_UP` when a reset occurs mid-warm-up (`heartbeat_count` ∈ [1, 99]). Distinguishes a disrupted session from a fresh power-on. |
| `0xA` | `ERR_LOW_BATTERY` *(advisory)* | Non-sticky; `vbatt_warn_s=1`. Battery below soft threshold; device still runs. Auto-clears when `vbatt_warn` deasserts. |
| `0xB` | `ERR_POWER_FAIL` *(advisory)* | Non-sticky; `vbatt_ok_s=0`. Battery below hard threshold; `start` is blocked. Running batch is not aborted. Auto-clears when `vbatt_ok` reasserts. |

**Sticky faults (0x1–0x7):** held until `rst_n` deasserted; always override advisory codes.  
**Advisory codes (0x8–0xB):** non-sticky; auto-clear on condition removal; do not block a running batch (except 0xB which blocks `start`).  
**Priority within sticky:** `ERR_SV_ZERO` > `ERR_SV_OVERFLOW` > `ERR_NUM_SAMPLES_ZERO` > `ERR_ILLEGAL_STATE` > `ERR_GAMMA_SAT` > `ERR_GAMMA_ZERO` > `ERR_FIFO_OVERFLOW`  
**Advisory check:** `error_code >= 4'h8` indicates an advisory (not a hard fault).

---

### Kernel Output Stream

| Signal | Dir (core) | Width | Description |
|--------|-----------|-------|-------------|
| `kernel_out` | output | 16 | Q6.10 RBF kernel value ∈ [0, 1] |
| `kernel_valid` | output | 1 | Held high until `kernel_ready` handshake completes |
| `kernel_ready` | input | 1 | MCU asserts to consume kernel word |

**Handshake:** `kernel_valid` is a set/clear latch (not a one-cycle pulse). The core holds it high until `kernel_ready` is asserted; the FSM advances on the rising edge of `kernel_ready`. This ensures the MCU cannot miss a kernel output even if it deasserts `kernel_ready` for multiple cycles.

**Output volume per batch:** `num_samples × Σ num_sv_per_class` kernel words  
**Output order:** sequential over (sample, class, sv_within_class) — innermost loop is sv

---

## svm_sv_ram_if — Core ↔ Support-Vector SRAM

| Signal | Dir (core) | Width | Description |
|--------|-----------|-------|-------------|
| `sv_ram_addr` | output | 18 | Word address |
| `sv_ram_ren` | output | 1 | Read enable; data valid one cycle later |
| `sv_ram_rdata` | input | 16 | Q6.10 SV feature word |

| Property | Value |
|----------|-------|
| Capacity | 250 SVs × 256 features × 2 B = 128 KB |
| Address space | 18-bit (2¹⁸ = 256 K words — sufficient) |
| Access | Read-only |
| Latency | 1 cycle (synchronous SRAM) |

---

## svm_work_ram_if — Core ↔ Workspace SRAM

| Signal | Dir (core) | Width | Description |
|--------|-----------|-------|-------------|
| `work_ram_addr` | output | 19 | Word address |
| `work_ram_wen` | output | 1 | Write enable |
| `work_ram_wdata` | output | 16 | Write data (Q6.10 distance or kernel) |
| `work_ram_ren` | output | 1 | Read enable |
| `work_ram_rdata` | input | 16 | Read data |

| Property | Value |
|----------|-------|
| Capacity | 1000 samples × 250 SVs × 2 B = 500 KB max |
| Address space | 19-bit (2¹⁹ = 512 K words — sufficient) |
| Access | Read/write |
| Note | Promoted from 18-bit (m2) to 19-bit (m3) to cover full 500 KB range |

---

## RTL Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_WIDTH` | 16 | Fixed-point word width (Q6.10) |
| `FRAC_BITS` | 10 | Fractional bits |
| `DIST_WIDTH` | 20 | Accumulator width for squared distance |
| `FEATURE_DIM` | 256 | Features per heartbeat |
| `NUM_SV` | 250 | Maximum support vectors |
| `MAX_BATCH_SIZE` | 1000 | Maximum heartbeats per batch |
| `FIFO_DEPTH` | 8192 | Input FIFO depth (words) |
| `ADDR_WIDTH` | 13 | Internal FIFO address bits |
| `DEFAULT_GAMMA` | 0.25 | γ reset value (Q6.10 = 256 = `0x0100`) |
| `DEFAULT_C` | 1.0 | C reset value (Q6.10 = 1024 = `0x0400`) |
| `DEFAULT_BIAS_[0:4]` | 0.0 | Per-class bias reset values |

---

## FSM States

| State | Description | Transition |
|-------|-------------|------------|
| `IDLE` | Waiting for `start` pulse | → `LOAD_FIFO` on `start && vbatt_ok_s`; blocked when `vbatt_ok=0` |
| `LOAD_FIFO` | Accumulating QSPI data in FIFO | → `LOAD_FEATURES` when `fifo_count ≥ FEATURE_DIM` |
| `LOAD_FEATURES` | Draining FIFO into feature registers | → `COMPUTE_DIST` after `FEATURE_DIM` words |
| `COMPUTE_DIST` | Squared-distance accumulation (all dims) | → `COMPUTE_KERNEL` on `dist_done` |
| `COMPUTE_KERNEL` | 15th-order Horner eval of exp(−γd²) | → `OUTPUT_RESULT` on `horner_done` |
| `OUTPUT_RESULT` | Stream kernel; handshake with MCU | → `COMPUTE_DIST` (next SV/class) · `LOAD_FIFO` (next heartbeat) · `IDLE` (batch done) |
| `ERROR_STATE` | One-cycle pass-through; latch error flag | → `IDLE` |

`COMPUTE_DIST → COMPUTE_KERNEL → OUTPUT_RESULT` repeats once per (SV, class) pair within a heartbeat. The outer loop over heartbeats returns to `LOAD_FIFO` between samples.

---

## m2 → m3 Interface Changes

| Item | m2 | m3 | Reason |
|------|----|----|--------|
| `param_addr` width | 2-bit | **3-bit** | Register map expanded for 5 bias registers |
| `bias_reg[5]` | Not present | **Added** | Per-class decision bias; programmable via param interface |
| `error_code[3:0]` | Not present | **Added** | Fault code: 7 sticky (0x1–0x7) + 4 advisory (0x8–0xB) |
| Default γ | 0.01 (`0x000A`) | **0.25 (`0x0100`)** | Matches sklearn-trained model parameters |
| `ERR_GAMMA_ZERO` | Not present | **Added (`0x6`)** | Detects γ=0 silently producing all-1.0 kernels |
| `vbatt_warn` / `vbatt_ok` | Not present | **Added** | Async comparator inputs; 2-FF synchronized inside core |
| Advisory codes 0x8–0xB | Not present | **Added** | Warm-up, interrupted, low-battery, power-fail advisories |
| `work_ram` `ADDR_WIDTH` | 18-bit | **19-bit** | 500 KB workspace requires 19 bits (2¹⁸ = 256 KB < 500 KB) |

## RTL Fixes Applied (m2 → m3, ASIC-ready)

| # | Fix | Symptom fixed |
|---|-----|---------------|
| 1 | `kernel_valid` hold register — changed 1-cycle pulse to set/clear latch held until `kernel_ready` | FSM stalled permanently when `kernel_ready=0` during the one cycle `kernel_valid` was asserted |
| 2 | `gamma_latched` shadow register — γ captured from `gamma_int` at `start`; Horner engine uses shadow throughout batch | Mid-compute `param_write_en` could corrupt in-flight kernel values |
| 3 | `ERR_GAMMA_ZERO` (`0x6`) — fires when `gamma_int == 0` while FSM is not `IDLE` | γ=0 silently produced all-1.0 kernels with no error raised |
| 4 | `ERR_WARMING_UP` (`0x8`) — non-sticky advisory; fires on clean start; auto-clears at beat 100 | Host had no signal that early results (beats 1–99) are unreliable during cold-start warm-up |
| 5 | `ERR_INTERRUPTED` (`0x9`) — non-sticky advisory; fires when `rst_n` pulse occurs mid-warm-up (`heartbeat_count` ∈ [1, 99]) | Host could not distinguish fresh power-on from a disrupted warm-up — both looked identical via `ERR_WARMING_UP` alone |
| 6 | `ERR_LOW_BATTERY` (`0xA`) / `ERR_POWER_FAIL` (`0xB`) — two new input pins `vbatt_warn` / `vbatt_ok`; 0xA advisory while running; 0xB blocks `start` | No hardware signal was available to warn the host MCU of low battery or prevent starting a classification without sufficient power |
| 7 | `num_samples_latched` shadow register — captured from `num_samples` at `start && vbatt_ok_s`; `last_heartbeat` uses the latched copy | A mid-batch `num_samples` write could corrupt batch-end detection, causing early termination or an infinite loop |
| 8 | `vbatt_ok_s` guard in IDLE — `sv_count_reg`, `gamma_latched`, and `num_samples_latched` only latch when `start && vbatt_ok_s` | IDLE counter block could capture stale values when `vbatt_ok=0` would have blocked the FSM from leaving IDLE anyway |
| 9 | 2-FF input synchronizers (`sync_ff` module) for `vbatt_ok` and `vbatt_warn` — reset values: `vbatt_ok→1`, `vbatt_warn→0`; FSM uses `_s` suffix signals | Async comparator outputs driven into a synchronous FSM violate setup/hold, causing metastability at netlist/ASIC |
| 10 | Distance matrix drain flush — 2 extra cycles after last `valid_in` flush the 2-stage `diff→diff_sq→accumulator` pipeline; `diff`/`diff_sq` reset in IDLE | Last 2 of 256 feature dimensions were silently dropped every kernel computation; `diff`/`diff_sq` persisted across SV computations, corrupting SV 2–N |
| 11 | `arm_interrupted` ASIC reset — `` `ifdef SYNTHESIS `` adds `negedge rst_n` async reset path; Icarus uses gated-only path to avoid its non-standard cross-block NBA ordering; inline `= 1'b0` is simulation init only (synthesis tools ignore it) | Without this fix synthesis produces an unreset FF (lint DRC violation); the inline init is not a reset and is silently dropped by Yosys/DC/Genus |

---

## Instantiation

Icarus Verilog does not support interface types in module port lists. Use flat signals:

```systemverilog
logic clk, rst_n;
logic        param_write_en;
logic [2:0]  param_addr;
logic [15:0] param_data, gamma_reg, c_reg;
logic [15:0] bias_reg [5];
logic [7:0]  num_sv_per_class [5];
logic        vbatt_warn, vbatt_ok;
logic        qspi_valid, qspi_ready;
logic [15:0] qspi_data;
logic        start, done, error;
logic [3:0]  error_code;
logic [9:0]  num_samples;
logic [15:0] kernel_out;
logic        kernel_valid, kernel_ready;
logic [17:0] sv_ram_addr;
logic [15:0] sv_ram_rdata;
logic        sv_ram_ren;
logic [18:0] work_ram_addr;  // 19-bit: covers 500 KB workspace
logic [15:0] work_ram_wdata, work_ram_rdata;
logic        work_ram_wen, work_ram_ren;

svm_compute_core dut (
    .clk(clk),                 .rst_n(rst_n),
    .param_write_en(param_write_en),
    .param_addr(param_addr),   .param_data(param_data),
    .gamma_reg(gamma_reg),     .c_reg(c_reg),
    .bias_reg(bias_reg),       .num_sv_per_class(num_sv_per_class),
    .vbatt_warn(vbatt_warn),   .vbatt_ok(vbatt_ok),
    .qspi_valid(qspi_valid),   .qspi_data(qspi_data),
    .qspi_ready(qspi_ready),
    .sv_ram_addr(sv_ram_addr), .sv_ram_rdata(sv_ram_rdata),
    .sv_ram_ren(sv_ram_ren),
    .work_ram_addr(work_ram_addr),
    .work_ram_wdata(work_ram_wdata), .work_ram_rdata(work_ram_rdata),
    .work_ram_wen(work_ram_wen),     .work_ram_ren(work_ram_ren),
    .start(start),             .num_samples(num_samples),
    .done(done),               .error(error),
    .error_code(error_code),
    .kernel_out(kernel_out),   .kernel_valid(kernel_valid),
    .kernel_ready(kernel_ready)
);
```

**Compile and run the interface testbench:**

```bash
iverilog -g2012 -o tb_if tb_interface.sv svm_compute_core.sv
vvp tb_if
# Output written to tb_interface.log
```
