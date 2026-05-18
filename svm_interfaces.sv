// ===========================================================================
// SVM Compute Core — SystemVerilog Interface Definitions  (m3, pre-netlist)
// ===========================================================================
// Three interfaces span the three physical boundaries of svm_compute_core:
//
//   svm_host_if      — MCU ↔ Core  (QSPI feature stream, parameter writes,
//                                    control/status, kernel output stream)
//   svm_sv_ram_if    — Core ↔ Support-Vector SRAM  (128 KB, read-only)
//   svm_work_ram_if  — Core ↔ Workspace SRAM       (≤500 KB, read/write)
//
// Updated from m2 to match the verified svm_compute_core.sv (19/19 tests PASS):
//   • param_addr widened 2→3 bits to address 5 bias registers
//   • bias_reg[5] output added (one per class, Q6.10)
//   • error_code[3:0] output added (sticky fault code, 7 defined values)
//   • DEFAULT_GAMMA corrected to 0.25 (was 0.01; Q6.10 = 256 = 0x0100)
//   • ERR_GAMMA_ZERO (0x6) added to error table
//
// ===========================================================================
// QSPI PROTOCOL
// ===========================================================================
//
//   Mode        : SPI Mode 0  (CPOL=0, CPHA=0)
//   Lanes       : 4 (quad SPI)
//   Clock       : 4 MHz SCK  →  16 Mbps  →  1 M words/sec  →  2 MB/s
//   Word width  : 16-bit Q6.10
//   Word framing: 4 SCK cycles per word, MSB first, IO[3] = MSB of each nibble
//
//     SCK cycle :  1            2            3            4
//     IO[3:0]   :  bits[15:12]  bits[11:8]   bits[7:4]    bits[3:0]
//                  captured on rising SCK (CPHA=0)
//
//   One heartbeat = 256 features = 256 µs at 4 MHz QSPI.
//   The QSPI deserializer is external; internally the core sees only the
//   qspi_valid/qspi_data/qspi_ready ready-valid bus.
//
// ===========================================================================
// REGISTER MAP  (param_addr [2:0] / param_data [15:0])
// ===========================================================================
//
//   Access: assert param_write_en for one cycle; param_addr selects register.
//   All registers are combinationally readable via their output signals.
//
//   Addr  Register      Width   Format  Reset default        Notes
//   ─────────────────────────────────────────────────────────────────────────
//   3'h0  gamma_reg     16-bit  Q6.10   256  (0.25)          RBF bandwidth γ
//   3'h1  c_reg         16-bit  Q6.10   1024 (1.0)           SVM penalty C
//   3'h2  bias_reg[0]   16-bit  Q6.10   0                    Class 0 bias
//   3'h3  bias_reg[1]   16-bit  Q6.10   0                    Class 1 bias
//   3'h4  bias_reg[2]   16-bit  Q6.10   0                    Class 2 bias
//   3'h5  bias_reg[3]   16-bit  Q6.10   0                    Class 3 bias
//   3'h6  bias_reg[4]   16-bit  Q6.10   0                    Class 4 bias
//   3'h7  (reserved)    —       —       —                    writes ignored
//
//   Q6.10 encoding:  raw = round(real_value × 1024)
//     γ = 0.25  →  raw = 256   = 0x0100  (exact)
//     C = 1.0   →  raw = 1024  = 0x0400  (exact)
//     Range: −32.000 to +31.999;  LSB ≈ 0.000977
//
//   GAMMA saturation threshold: 0x2000 (8192 Q6.10 ≈ 8.0).
//   Writes above this threshold set ERR_GAMMA_SAT without updating gamma_reg.
//
// ===========================================================================
// ERROR CODE TABLE  (error_code [3:0], sticky, cleared only by rst_n)
// ===========================================================================
//
//   Code  Name               Trigger condition
//   ────────────────────────────────────────────────────────────────────────
//   0x0   ERR_NONE           No fault
//   0x1   ERR_SV_ZERO        Σ num_sv_per_class = 0 at start
//   0x2   ERR_SV_OVERFLOW    Σ num_sv_per_class > 250 at start
//   0x3   ERR_ILLEGAL_STATE  FSM default branch taken (internal fault)
//   0x4   ERR_GAMMA_SAT      gamma_int > 8192 during param write
//   0x5   ERR_FIFO_OVERFLOW  QSPI data arrived while FIFO full (data dropped)
//   0x6   ERR_GAMMA_ZERO     gamma_int = 0 while FSM not IDLE
//                            (all kernels collapse to 1.0 — silent classifier
//                             failure; error flag forces safe shutdown)
//
//   Priority (highest first): ERR_SV_ZERO > ERR_SV_OVERFLOW >
//     ERR_ILLEGAL_STATE > ERR_GAMMA_SAT > ERR_GAMMA_ZERO > ERR_FIFO_OVERFLOW
//
// ===========================================================================
// FSM STATE SUMMARY
// ===========================================================================
//
//   IDLE          → LOAD_FIFO       on start pulse (if no error)
//   LOAD_FIFO     → LOAD_FEATURES   when FIFO holds ≥ 256 words
//   LOAD_FEATURES → COMPUTE_DIST    after 256 feature words consumed
//   COMPUTE_DIST  → COMPUTE_KERNEL  when distance_matrix asserts dist_done
//   COMPUTE_KERNEL→ OUTPUT_RESULT   when Horner engine asserts horner_done
//   OUTPUT_RESULT → LOAD_FIFO       if more heartbeats remain in batch
//              └──→ IDLE            after last heartbeat (done asserted)
//   ERROR_STATE   → IDLE            (one-cycle pass-through; error flag sticky)
//
// ===========================================================================


// ---------------------------------------------------------------------------
// svm_host_if
// MCU ↔ svm_compute_core boundary.
// ---------------------------------------------------------------------------
interface svm_host_if #(
    parameter int DATA_WIDTH = 16
) (
    input logic clk,
    input logic rst_n
);

    // --- Parameter programming ---
    logic                    param_write_en;
    logic [2:0]              param_addr;        // 3-bit: addresses gamma, C, bias[0:4]
    logic [DATA_WIDTH-1:0]   param_data;
    logic [DATA_WIDTH-1:0]   gamma_reg;         // readback
    logic [DATA_WIDTH-1:0]   c_reg;             // readback
    logic [DATA_WIDTH-1:0]   bias_reg [5];      // readback, one per class

    // --- Per-class SV counts (set before asserting start) ---
    logic [7:0]              num_sv_per_class [5];

    // --- QSPI feature stream (deserialized, ready-valid) ---
    logic                    qspi_valid;
    logic [DATA_WIDTH-1:0]   qspi_data;
    logic                    qspi_ready;        // core asserts when FIFO not full

    // --- Batch control ---
    logic                    start;             // one-cycle pulse, IDLE only
    logic [9:0]              num_samples;       // heartbeats in batch (1–1000)

    // --- Status ---
    logic                    done;              // one-cycle pulse after last kernel
    logic                    error;             // sticky flag
    logic [3:0]              error_code;        // latched fault code (see table above)

    // --- Kernel output stream (ready-valid) ---
    logic [DATA_WIDTH-1:0]   kernel_out;
    logic                    kernel_valid;      // core drives; held until kernel_ready
    logic                    kernel_ready;      // MCU drives to consume kernel

    // MCU (host) modport
    modport host (
        output param_write_en,
        output param_addr,
        output param_data,
        input  gamma_reg,
        input  c_reg,
        input  bias_reg,
        output num_sv_per_class,
        output qspi_valid,
        output qspi_data,
        input  qspi_ready,
        output start,
        output num_samples,
        input  done,
        input  error,
        input  error_code,
        input  kernel_out,
        input  kernel_valid,
        output kernel_ready
    );

    // Core modport
    modport core (
        input  param_write_en,
        input  param_addr,
        input  param_data,
        output gamma_reg,
        output c_reg,
        output bias_reg,
        input  num_sv_per_class,
        input  qspi_valid,
        input  qspi_data,
        output qspi_ready,
        input  start,
        input  num_samples,
        output done,
        output error,
        output error_code,
        output kernel_out,
        output kernel_valid,
        input  kernel_ready
    );

endinterface : svm_host_if


// ---------------------------------------------------------------------------
// svm_sv_ram_if
// Core ↔ Support-Vector SRAM.
// 250 SVs × 256 features × 2 B = 128 KB  →  18-bit address, read-only.
// One-cycle read latency (synchronous SRAM model).
// ---------------------------------------------------------------------------
interface svm_sv_ram_if #(
    parameter int DATA_WIDTH = 16,
    parameter int ADDR_WIDTH = 18
) ();

    logic [ADDR_WIDTH-1:0]  addr;
    logic [DATA_WIDTH-1:0]  rdata;
    logic                   ren;

    modport core (
        output addr,
        output ren,
        input  rdata
    );

    modport ram (
        input  addr,
        input  ren,
        output rdata
    );

endinterface : svm_sv_ram_if


// ---------------------------------------------------------------------------
// svm_work_ram_if
// Core ↔ Workspace SRAM.
// 1000 samples × 250 SVs × 2 B = 500 KB max  →  19-bit address, R/W.
// Note: ADDR_WIDTH promoted to 19 bits (from 18) to cover full 500 KB range.
// ---------------------------------------------------------------------------
interface svm_work_ram_if #(
    parameter int DATA_WIDTH = 16,
    parameter int ADDR_WIDTH = 19
) ();

    logic [ADDR_WIDTH-1:0]  addr;
    logic [DATA_WIDTH-1:0]  wdata;
    logic [DATA_WIDTH-1:0]  rdata;
    logic                   wen;
    logic                   ren;

    modport core (
        output addr,
        output wdata,
        input  rdata,
        output wen,
        output ren
    );

    modport ram (
        input  addr,
        input  wdata,
        output rdata,
        input  wen,
        input  ren
    );

endinterface : svm_work_ram_if


// ===========================================================================
// Instantiation guide — flat port wiring for Icarus Verilog 13 testbenches
// (Icarus does not support interface types in module port lists; use flat
//  signals and connect directly.)
//
//   logic clk, rst_n;
//   logic        param_write_en;
//   logic [2:0]  param_addr;
//   logic [15:0] param_data, gamma_reg, c_reg;
//   logic [15:0] bias_reg[5];
//   logic [7:0]  num_sv_per_class[5];
//   logic        qspi_valid, qspi_ready;
//   logic [15:0] qspi_data;
//   logic        start, done, error;
//   logic [3:0]  error_code;
//   logic [9:0]  num_samples;
//   logic [15:0] kernel_out;
//   logic        kernel_valid, kernel_ready;
//   logic [17:0] sv_ram_addr;  logic [15:0] sv_ram_rdata;  logic sv_ram_ren;
//   logic [18:0] work_ram_addr; logic [15:0] work_ram_wdata, work_ram_rdata;
//   logic        work_ram_wen, work_ram_ren;
//
//   svm_compute_core dut (
//       .clk(clk), .rst_n(rst_n),
//       .param_write_en(param_write_en), .param_addr(param_addr),
//       .param_data(param_data), .gamma_reg(gamma_reg), .c_reg(c_reg),
//       .bias_reg(bias_reg), .num_sv_per_class(num_sv_per_class),
//       .qspi_valid(qspi_valid), .qspi_data(qspi_data), .qspi_ready(qspi_ready),
//       .sv_ram_addr(sv_ram_addr), .sv_ram_rdata(sv_ram_rdata), .sv_ram_ren(sv_ram_ren),
//       .work_ram_addr(work_ram_addr), .work_ram_wdata(work_ram_wdata),
//       .work_ram_rdata(work_ram_rdata), .work_ram_wen(work_ram_wen),
//       .work_ram_ren(work_ram_ren),
//       .start(start), .num_samples(num_samples), .done(done),
//       .error(error), .error_code(error_code),
//       .kernel_out(kernel_out), .kernel_valid(kernel_valid), .kernel_ready(kernel_ready)
//   );
// ===========================================================================
