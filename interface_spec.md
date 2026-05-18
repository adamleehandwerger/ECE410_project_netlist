# Interface Specification — SVM Compute Core (m3, Pre-Netlist Verified)

**RTL file:** `compute_core/svm_compute_core.sv`  
**Interface file:** `compute_core/svm_interfaces.sv`  
**Verification status:** 19/19 tests PASS (10 iverilog + 9 cocotb)  
**Date:** 2026-05-07

---

## Changes from m2

| Item | m2 | m3 | Reason |
|------|----|----|--------|
| `param_addr` width | 2-bit | **3-bit** | Register map expanded to include 5 bias registers |
| `bias_reg[5]` | Not present | **Added** | Per-class decision bias; programmable via param interface |
| `error_code[3:0]` | Not present | **Added** | Sticky fault code distinguishes 7 error conditions |
| Default γ | 0.01 (= 10 Q6.10) | **0.25 (= 256 = 0x0100)** | Matches sklearn-trained model parameters |
| `ERR_GAMMA_ZERO` | Not present | **Added (0x6)** | Detects γ=0 silently producing all-1.0 kernels |
| `work_ram` ADDR_WIDTH | 18-bit | **19-bit** | 500 KB workspace needs 19 bits (2^18 = 256 KB < 500 KB) |

---

## Physical Boundary Overview

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

---

## svm_host_if — MCU ↔ Core

### QSPI Feature Stream

| Signal | Dir (core) | Width | Description |
|--------|-----------|-------|-------------|
| `qspi_valid` | input | 1 | Deserializer asserts when a 16-bit word is ready |
| `qspi_data` | input | 16 | Q6.10 feature word |
| `qspi_ready` | output | 1 | Core deasserts when input FIFO is full (8192-word depth) |

**Protocol:** Mode 0 QSPI (CPOL=0, CPHA=0), 4 data lanes, 4 MHz SCK  
**Throughput:** 16 Mbps → 2 MB/s → 1 M words/sec  
**One heartbeat:** 256 features → 256 µs transfer time  
**FIFO:** 8192 words (16 KB); core backpressures via `qspi_ready`

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

| Address | Register | Default | Q6.10 value | Notes |
|---------|----------|---------|-------------|-------|
| `3'h0` | `gamma_reg` | `0x0100` | 0.25 | RBF bandwidth γ; saturates at 0x2000 |
| `3'h1` | `c_reg` | `0x0400` | 1.0 | SVM penalty C |
| `3'h2` | `bias_reg[0]` | `0x0000` | 0.0 | Class 0 (Normal) decision bias |
| `3'h3` | `bias_reg[1]` | `0x0000` | 0.0 | Class 1 (PVC) decision bias |
| `3'h4` | `bias_reg[2]` | `0x0000` | 0.0 | Class 2 (AFib) decision bias |
| `3'h5` | `bias_reg[3]` | `0x0000` | 0.0 | Class 3 (VT) decision bias |
| `3'h6` | `bias_reg[4]` | `0x0000` | 0.0 | Class 4 (SVT) decision bias |
| `3'h7` | (reserved) | — | — | Writes ignored |

**Q6.10 encoding:** `raw = round(real_value × 1024)`  
Range: −32.000 to +31.999, LSB ≈ 0.000977  
γ saturation threshold: `0x2000` (8.0); writes above this set `ERR_GAMMA_SAT`

---

### Batch Control

| Signal | Dir (core) | Width | Description |
|--------|-----------|-------|-------------|
| `num_sv_per_class[5]` | input | 8×5 | SV count per class; latched on `start`; sum must be 1–250 |
| `start` | input | 1 | One-cycle pulse; valid in IDLE state only |
| `num_samples` | input | 10 | Heartbeats in this batch (1–1000); latched on `start` |
| `done` | output | 1 | One-cycle pulse after the last kernel output of the batch |

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
| `0x2` | `ERR_SV_OVERFLOW` | Σ `num_sv_per_class` > 250 at `start` |
| `0x3` | `ERR_ILLEGAL_STATE` | FSM default branch taken (internal fault) |
| `0x4` | `ERR_GAMMA_SAT` | `gamma_int` > 8192 during param write |
| `0x5` | `ERR_FIFO_OVERFLOW` | QSPI data arrived when FIFO full (data dropped) |
| `0x6` | `ERR_GAMMA_ZERO` | `gamma_int` = 0 while FSM not IDLE (silent classifier failure) |

Priority (highest wins): `ERR_SV_ZERO` > `ERR_SV_OVERFLOW` > `ERR_ILLEGAL_STATE` > `ERR_GAMMA_SAT` > `ERR_GAMMA_ZERO` > `ERR_FIFO_OVERFLOW`

---

### Kernel Output Stream

| Signal | Dir (core) | Width | Description |
|--------|-----------|-------|-------------|
| `kernel_out` | output | 16 | Q6.10 RBF kernel value ∈ [0, 1] |
| `kernel_valid` | output | 1 | Held high until `kernel_ready` handshake completes (Fix 1) |
| `kernel_ready` | input | 1 | MCU asserts to consume kernel; core advances on rising edge |

**Throughput:** One kernel per SV per heartbeat  
**Total outputs per batch:** `num_samples × Σ num_sv_per_class` kernel words

---

## svm_sv_ram_if — Core ↔ Support-Vector SRAM

| Signal | Dir (core) | Width | Description |
|--------|-----------|-------|-------------|
| `sv_ram_addr` | output | 18 | Word address into SV SRAM |
| `sv_ram_ren` | output | 1 | Read enable; data valid one cycle later |
| `sv_ram_rdata` | input | 16 | Q6.10 SV feature word |

**Capacity:** 250 SVs × 256 features × 2 B = 128 KB  
**Address space:** 18-bit (2^18 = 256 K words — sufficient)  
**Latency:** 1 cycle (synchronous SRAM model)

---

## svm_work_ram_if — Core ↔ Workspace SRAM

| Signal | Dir (core) | Width | Description |
|--------|-----------|-------|-------------|
| `work_ram_addr` | output | 19 | Word address into workspace SRAM |
| `work_ram_wen` | output | 1 | Write enable |
| `work_ram_wdata` | output | 16 | Write data (Q6.10 distance or kernel) |
| `work_ram_ren` | output | 1 | Read enable |
| `work_ram_rdata` | input | 16 | Read data |

**Capacity:** 1000 samples × 250 SVs × 2 B = 500 KB  
**Address space:** 19-bit (2^19 = 512 K words — sufficient)  
**Note:** m2 used 18-bit; promoted to 19-bit to cover full 500 KB range

---

## RTL Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DATA_WIDTH` | 16 | Fixed-point word width (Q6.10) |
| `FRAC_BITS` | 10 | Fractional bits |
| `DIST_WIDTH` | 20 | Accumulator width for squared distance |
| `FEATURE_DIM` | 256 | Features per heartbeat |
| `NUM_SV` | 250 | Maximum support vectors |
| `MAX_BATCH_SIZE` | 1000 | Maximum heartbeats per batch |
| `FIFO_DEPTH` | 8192 | Input FIFO depth (words) |
| `ADDR_WIDTH` | 13 | Internal FIFO address bits |
| `DEFAULT_GAMMA` | 0.25 | γ reset value (Q6.10 = 256 = 0x0100) |
| `DEFAULT_C` | 1.0 | C reset value (Q6.10 = 1024 = 0x0400) |
| `DEFAULT_BIAS_[0:4]` | 0.0 | Per-class bias reset values |

---

## FSM States

| State | Description | Next state condition |
|-------|-------------|----------------------|
| `IDLE` | Waiting for `start` pulse | → `LOAD_FIFO` on `start` |
| `LOAD_FIFO` | Accumulating QSPI data in FIFO | → `LOAD_FEATURES` when FIFO ≥ 256 words |
| `LOAD_FEATURES` | Reading features from FIFO to working registers | → `COMPUTE_DIST` after 256 words |
| `COMPUTE_DIST` | Sequential squared-distance accumulation over all 256 dims | → `COMPUTE_KERNEL` on `dist_done` |
| `COMPUTE_KERNEL` | 15th-order Horner evaluation of exp(−γd²) | → `OUTPUT_RESULT` on `horner_done` |
| `OUTPUT_RESULT` | Streaming kernel value; await `kernel_ready` | → `LOAD_FIFO` (more heartbeats) or `IDLE` (batch done) |
| `ERROR_STATE` | One-cycle pass-through; `error` flag latched sticky | → `IDLE` |

---

## RTL Fixes Applied (m2 → m3)

| # | Fix | Symptom fixed |
|---|-----|---------------|
| 1 | `kernel_valid` hold register — changed 1-cycle pulse to set/clear latch held until `kernel_ready` | FSM stalled permanently when `kernel_ready=0` during the one cycle `kernel_valid` was high |
| 2 | `gamma_latched` shadow register — γ captured from `gamma_int` at `start`; Horner engine uses shadow throughout batch | Mid-compute `param_write_en` could corrupt in-flight kernel values |
| 3 | `ERR_GAMMA_ZERO` (0x6) — fires when `gamma_int == 0` while FSM is not IDLE | γ=0 silently produced all-1.0 kernels with no error raised |
