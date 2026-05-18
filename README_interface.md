# SVM Compute Core — Interface Reference

**RTL:** `svm_compute_core.sv`  
**Interface definitions:** `svm_interfaces.sv`  
**Testbench:** `tb_interface.sv` → `tb_interface.log`  
**Verification status:** 19/19 unit tests PASS · 24/24 interface checks PASS  
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

### Batch Control

| Signal | Dir (core) | Width | Description |
|--------|-----------|-------|-------------|
| `num_sv_per_class[5]` | input | 8×5 | SV count per class; evaluated at `start` |
| `start` | input | 1 | One-cycle pulse; valid in IDLE state only |
| `num_samples` | input | 10 | Heartbeats in this batch (1–1000) |
| `done` | output | 1 | One-cycle pulse after the last kernel output |

**`num_samples` is a live wire**, not latched at `start`. The FSM evaluates `sample_counter >= num_samples - 1` continuously. Changing `num_samples` mid-batch will affect when `done` fires.

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

**Priority (highest wins):** `ERR_SV_ZERO` > `ERR_SV_OVERFLOW` > `ERR_NUM_SAMPLES_ZERO` > `ERR_ILLEGAL_STATE` > `ERR_GAMMA_SAT` > `ERR_GAMMA_ZERO` > `ERR_FIFO_OVERFLOW`

Both `error` and `error_code` are sticky — they hold their values until `rst_n` is deasserted. A batch that raises an error does not produce valid kernel output.

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
| `IDLE` | Waiting for `start` pulse | → `LOAD_FIFO` on valid `start` |
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
| `error_code[3:0]` | Not present | **Added** | Sticky fault code distinguishing 7 error conditions |
| Default γ | 0.01 (`0x000A`) | **0.25 (`0x0100`)** | Matches sklearn-trained model parameters |
| `ERR_GAMMA_ZERO` | Not present | **Added (`0x6`)** | Detects γ=0 silently producing all-1.0 kernels |
| `work_ram` `ADDR_WIDTH` | 18-bit | **19-bit** | 500 KB workspace requires 19 bits (2¹⁸ = 256 KB < 500 KB) |

## RTL Fixes Applied (m2 → m3)

| # | Fix | Symptom fixed |
|---|-----|---------------|
| 1 | `kernel_valid` hold register — changed 1-cycle pulse to set/clear latch held until `kernel_ready` | FSM stalled permanently when `kernel_ready=0` during the one cycle `kernel_valid` was asserted |
| 2 | `gamma_latched` shadow register — γ captured from `gamma_int` at `start`; Horner engine uses shadow throughout batch | Mid-compute `param_write_en` could corrupt in-flight kernel values |
| 3 | `ERR_GAMMA_ZERO` (`0x6`) — fires when `gamma_int == 0` while FSM is not `IDLE` | γ=0 silently produced all-1.0 kernels with no error raised |

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
