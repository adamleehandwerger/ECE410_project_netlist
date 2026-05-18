// ============================================================================
// Multi-Class Cardiac Arrhythmia Detection — SVM Compute Core  (LUT kernel)
// ECE 410 Project
// ============================================================================
//
// OVERVIEW
// --------
// Four modules implement a duty-cycling RBF-SVM classifier for five cardiac
// arrhythmia classes (Normal, PVC, AFib, VT, SVT).
//
//   svm_compute_core  — top-level FSM; integrates the three submodules below
//   input_fifo        — 8192-word (16 KB) synchronous FIFO; buffers QSPI data
//   distance_matrix   — squared Euclidean distance Σ(x[k]-sv[k])²
//   horner_engine     — range-reduction LUT + 15th-order Horner for exp(-γD)
//
// RANGE-REDUCTION LUT KERNEL  (replaces single-stage Horner from _updated)
// -------------------------------------------------------------------------
// Problem: at gamma=0.25 the product γ·D can reach ~16, causing int16
//          overflow in the naive Horner: result wraps to 1.0 instead of ~0,
//          collapsing the classifier to near-random (≈20% accuracy).
//
// Fix: exp(-γ·D) = exp(-I) × exp(-F)
//   P   = γ × D  in Q6.10  (held as 36-bit unsigned product)
//   I   = P >> 20           (integer part; 0–15 valid; ≥16 → result = 0)
//   F_q = (P >> 10) & 0x3FF (fractional part in Q6.10, always ∈ [0,1))
//
//   exp(-I)   : 16-entry read-only LUT (EXP_INT_LUT), Q6.10 values
//               [1024, 377, 139, 51, 19, 7, 3, 1, 0×8 entries]
//   exp(-F_q) : existing Horner polynomial with x = -F_q ∈ [-1023, 0]
//               (always in valid range → full accuracy)
//   result    : (LUT_val × Horner_val) >> FRAC_BITS, clamped to [0, 1024]
//
// Net effect: gamma=0.25 gives sklearn accuracy = HW accuracy = 98.00%
//             vs. 96.33% at old gamma=0.01.
//
// FIXED-POINT FORMAT
// ------------------
// All data words: Q6.10 (16-bit, 10 fractional bits).
// real_value = raw_integer / 1024.0
// gamma = 0.25 → raw = 256 = 0x0100 (exact representation)
//
// ============================================================================
// CLOCK / RESET / PORT reference: identical to ECE410_project_updated version.
// Only DEFAULT_GAMMA and horner_engine internals have changed.
// ============================================================================

module svm_compute_core #(
    parameter int DATA_WIDTH = 16,
    parameter int FRAC_BITS = 10,
    parameter int DIST_WIDTH = 20,
    parameter int FEATURE_DIM = 256,
    parameter int NUM_SV = 250,
    parameter int MAX_BATCH_SIZE = 1000,
    parameter int FIFO_DEPTH = 8192,
    parameter int ADDR_WIDTH = 13,
    parameter real DEFAULT_GAMMA   = 0.25,   // Q6.10 = 256 = 0x0100 (exact)
    parameter real DEFAULT_C       = 1.0,
    parameter real DEFAULT_BIAS_0  =  0.0,
    parameter real DEFAULT_BIAS_1  =  0.0,
    parameter real DEFAULT_BIAS_2  =  0.0,
    parameter real DEFAULT_BIAS_3  =  0.0,
    parameter real DEFAULT_BIAS_4  =  0.0
) (
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic                    param_write_en,
    input  logic [2:0]              param_addr,
    input  logic [DATA_WIDTH-1:0]   param_data,
    output logic [DATA_WIDTH-1:0]   gamma_reg,
    output logic [DATA_WIDTH-1:0]   c_reg,
    output logic [DATA_WIDTH-1:0]   bias_reg [5],

    input  logic [7:0]              num_sv_per_class [5],

    input  logic                    qspi_valid,
    input  logic [DATA_WIDTH-1:0]   qspi_data,
    output logic                    qspi_ready,

    output logic [17:0]             sv_ram_addr,
    input  logic [DATA_WIDTH-1:0]   sv_ram_rdata,
    output logic                    sv_ram_ren,

    output logic [18:0]             work_ram_addr,
    output logic [DATA_WIDTH-1:0]   work_ram_wdata,
    input  logic [DATA_WIDTH-1:0]   work_ram_rdata,
    output logic                    work_ram_wen,
    output logic                    work_ram_ren,

    input  logic                    vbatt_warn,  // battery below soft threshold (warn; still operational)
    input  logic                    vbatt_ok,    // battery above hard threshold (sufficient for compute)

    input  logic                    start,
    input  logic [9:0]              num_samples,
    output logic                    done,
    output logic                    error,
    output logic [3:0]              error_code,

    output logic [DATA_WIDTH-1:0]   kernel_out,
    output logic                    kernel_valid,
    input  logic                    kernel_ready
);

    // =========================================================================
    // Constants
    // =========================================================================

    // ── Error codes ───────────────────────────────────────────────────────────
    localparam logic [3:0] ERR_NONE          = 4'h0; // no fault
    localparam logic [3:0] ERR_SV_ZERO       = 4'h1; // Σsv_count = 0
    localparam logic [3:0] ERR_SV_OVERFLOW   = 4'h2; // Σsv_count > NUM_SV
    localparam logic [3:0] ERR_ILLEGAL_STATE = 4'h3; // FSM default branch taken
    localparam logic [3:0] ERR_GAMMA_SAT     = 4'h4; // gamma > saturation threshold
    localparam logic [3:0] ERR_FIFO_OVERFLOW = 4'h5; // QSPI data dropped (FIFO full)
    localparam logic [3:0] ERR_GAMMA_ZERO    = 4'h6; // gamma = 0 → all kernels = 1.0 (silent classifier failure)
    localparam logic [3:0] ERR_NUM_SAMPLES_ZERO = 4'h7; // num_samples = 0 → last_heartbeat underflows, batch never ends
    localparam logic [3:0] ERR_WARMING_UP       = 4'h8; // fewer than 100 heartbeats since reset — RR/mean features unreliable (non-sticky)
    localparam logic [3:0] ERR_INTERRUPTED      = 4'h9; // reset fired mid-warm-up (heartbeat_count was in [1,99]) — restart required (non-sticky)
    localparam logic [3:0] ERR_LOW_BATTERY      = 4'hA; // vbatt_warn asserted: battery below soft threshold; recharge soon (non-sticky advisory)
    localparam logic [3:0] ERR_POWER_FAIL       = 4'hB; // vbatt_ok deasserted: power too low to classify; start is blocked (non-sticky advisory)

    // gamma > 8.0 (Q6.10 = 8192) means exp(-I) = 0 for I>=8; all kernels zero
    localparam logic [DATA_WIDTH-1:0] GAMMA_SAT_THRESH = 16'd8192;

    localparam logic [DATA_WIDTH-1:0] GAMMA_DEFAULT =
        DATA_WIDTH'($rtoi(DEFAULT_GAMMA  * (2.0 ** FRAC_BITS)));
    localparam logic [DATA_WIDTH-1:0] C_DEFAULT =
        DATA_WIDTH'($rtoi(DEFAULT_C      * (2.0 ** FRAC_BITS)));
    localparam logic [DATA_WIDTH-1:0] BIAS0_DEFAULT =
        DATA_WIDTH'($rtoi(DEFAULT_BIAS_0 * (2.0 ** FRAC_BITS)));
    localparam logic [DATA_WIDTH-1:0] BIAS1_DEFAULT =
        DATA_WIDTH'($rtoi(DEFAULT_BIAS_1 * (2.0 ** FRAC_BITS)));
    localparam logic [DATA_WIDTH-1:0] BIAS2_DEFAULT =
        DATA_WIDTH'($rtoi(DEFAULT_BIAS_2 * (2.0 ** FRAC_BITS)));
    localparam logic [DATA_WIDTH-1:0] BIAS3_DEFAULT =
        DATA_WIDTH'($rtoi(DEFAULT_BIAS_3 * (2.0 ** FRAC_BITS)));
    localparam logic [DATA_WIDTH-1:0] BIAS4_DEFAULT =
        DATA_WIDTH'($rtoi(DEFAULT_BIAS_4 * (2.0 ** FRAC_BITS)));

    // =========================================================================
    // Internal Signals
    // =========================================================================

    logic [DATA_WIDTH-1:0] gamma_int;
    logic [DATA_WIDTH-1:0] gamma_latched; // shadow: captured at start, used by Horner
    logic [DATA_WIDTH-1:0] c_int;
    logic [DATA_WIDTH-1:0] bias_int [5];

    // Error diagnostics
    logic                    fifo_overflow_r; // sticky: FIFO full while QSPI valid
    logic [3:0]              err_detect;      // combinational priority encoder

    // FIFO
    logic                    fifo_wr_en;
    logic                    fifo_rd_en;
    logic [DATA_WIDTH-1:0]   fifo_wr_data;
    logic [DATA_WIDTH-1:0]   fifo_rd_data;
    logic                    fifo_full;
    logic                    fifo_empty;
    logic [ADDR_WIDTH:0]     fifo_count;

    // Distance Matrix
    logic                    dist_start;
    logic                    dist_done;
    logic [DATA_WIDTH-1:0]   dist_feature_in;
    logic [DATA_WIDTH-1:0]   dist_sv_in;
    logic                    dist_valid_in;
    logic [DIST_WIDTH-1:0]   dist_out;
    logic                    dist_valid_out;

    // Horner Engine (LUT version)
    logic                    horner_start;
    logic                    horner_done;
    logic [DIST_WIDTH-1:0]   horner_dist_in;
    logic                    horner_valid_in;
    logic [DATA_WIDTH-1:0]   horner_kernel_out;
    logic                    horner_valid_out;

    logic [6:0]              heartbeat_count; // saturates at 100; persists across batches, clears on rst_n
    logic                    interrupted;     // set on rst_n when heartbeat_count was in [1,99]; cleared at beat 100
    logic                    arm_interrupted = 1'b0; // gated mirror of (count in [1,99]); only updates when
                                                     // rst_n=1 so it freezes at the pre-reset value while
                                                     // the reset pulse is held low

    logic                    dist_valid_latch;

    // Input synchronizers (2-FF) for async analog comparator outputs
    logic                    vbatt_ok_s;   // synchronized vbatt_ok  (reset=1: assume power OK)
    logic                    vbatt_warn_s; // synchronized vbatt_warn (reset=0: no warning)

    logic [DATA_WIDTH-1:0]   feature_bank [FEATURE_DIM];

    logic [9:0]              num_samples_latched; // shadow: captured at start; immune to mid-batch writes

    logic [7:0]              feat_wr_addr;
    logic                    feat_wr_en_r;
    logic [7:0]              feat_wr_addr_r;
    logic [8:0]              feat_wr_count;

    logic [7:0]              feat_rd_addr;
    logic                    feat_rd_en;
    logic                    feat_rd_en_r;
    logic [DATA_WIDTH-1:0]   feat_rd_data;

    logic [9:0]              sv_base;
    logic [17:0]             kernel_out_counter;

    typedef enum logic [2:0] {
        IDLE,
        LOAD_FIFO,
        LOAD_FEATURES,
        COMPUTE_DIST,
        COMPUTE_KERNEL,
        OUTPUT_RESULT,
        ERROR_STATE
    } state_t;

    state_t state, next_state;

    logic [9:0]  sample_counter;
    logic [7:0]  sv_counter;
    logic [2:0]  class_counter;

    logic [7:0] sv_count_reg [5];

    logic [10:0] total_sv_check;
    assign total_sv_check = sv_count_reg[0] + sv_count_reg[1]
                          + sv_count_reg[2] + sv_count_reg[3]
                          + sv_count_reg[4];

    wire last_sv        = (sv_counter >= sv_count_reg[class_counter] - 1);
    wire last_class     = (class_counter >= 3'd4);
    wire last_heartbeat = (sample_counter >= num_samples_latched - 1);

    // =========================================================================
    // Sub-module Instances
    // =========================================================================

    input_fifo #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(FIFO_DEPTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) u_input_fifo (
        .clk(clk), .rst_n(rst_n),
        .wr_en(fifo_wr_en), .wr_data(fifo_wr_data),
        .rd_en(fifo_rd_en), .rd_data(fifo_rd_data),
        .full(fifo_full), .empty(fifo_empty), .count(fifo_count)
    );

    distance_matrix #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .DIST_WIDTH(DIST_WIDTH),
        .FEATURE_DIM(FEATURE_DIM)
    ) u_distance_matrix (
        .clk(clk), .rst_n(rst_n),
        .start(dist_start),
        .feature_in(dist_feature_in), .sv_in(dist_sv_in), .valid_in(dist_valid_in),
        .dist_out(dist_out), .valid_out(dist_valid_out), .done(dist_done)
    );

    // 2-FF synchronizers for vbatt pins (async inputs from analog comparators)
    sync_ff #(.RESET_VAL(1'b1)) u_sync_vbatt_ok (
        .clk(clk), .rst_n(rst_n), .d(vbatt_ok),   .q(vbatt_ok_s)
    );
    sync_ff #(.RESET_VAL(1'b0)) u_sync_vbatt_warn (
        .clk(clk), .rst_n(rst_n), .d(vbatt_warn), .q(vbatt_warn_s)
    );

    horner_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .DIST_WIDTH(DIST_WIDTH)
    ) u_horner_engine (
        .clk(clk), .rst_n(rst_n),
        .start(horner_start), .dist_in(horner_dist_in),
        .valid_in(horner_valid_in), .gamma(gamma_latched),
        .kernel_out(horner_kernel_out), .valid_out(horner_valid_out), .done(horner_done)
    );

    // =========================================================================
    // Parameter Registers
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gamma_int    <= GAMMA_DEFAULT;
            c_int        <= C_DEFAULT;
            bias_int[0]  <= BIAS0_DEFAULT;
            bias_int[1]  <= BIAS1_DEFAULT;
            bias_int[2]  <= BIAS2_DEFAULT;
            bias_int[3]  <= BIAS3_DEFAULT;
            bias_int[4]  <= BIAS4_DEFAULT;
        end else if (param_write_en) begin
            case (param_addr)
                3'b000: gamma_int   <= param_data;
                3'b001: c_int       <= param_data;
                3'b010: bias_int[0] <= param_data;
                3'b011: bias_int[1] <= param_data;
                3'b100: bias_int[2] <= param_data;
                3'b101: bias_int[3] <= param_data;
                3'b110: bias_int[4] <= param_data;
                default: begin end
            endcase
        end
    end

    assign gamma_reg = gamma_int;
    assign c_reg     = c_int;
    generate
        for (genvar i = 0; i < 5; i++)
            assign bias_reg[i] = bias_int[i];
    endgenerate

    // =========================================================================
    // FSM
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            IDLE:          if (start && vbatt_ok_s) next_state = LOAD_FIFO;
            LOAD_FIFO:     if (fifo_count >= FEATURE_DIM) next_state = LOAD_FEATURES;
            LOAD_FEATURES: if (feat_wr_count == FEATURE_DIM) next_state = COMPUTE_DIST;
            COMPUTE_DIST:  if (dist_done) next_state = COMPUTE_KERNEL;
            COMPUTE_KERNEL: if (horner_done) next_state = OUTPUT_RESULT;
            OUTPUT_RESULT: begin
                if (kernel_ready && kernel_valid) begin
                    if (last_sv && last_class) begin
                        if (last_heartbeat) next_state = IDLE;
                        else                next_state = LOAD_FIFO;
                    end else begin
                        next_state = COMPUTE_DIST;
                    end
                end
            end
            ERROR_STATE: next_state = IDLE;
            default:     next_state = ERROR_STATE;
        endcase
    end

    // =========================================================================
    // FIFO Control
    // =========================================================================

    always_comb begin
        fifo_wr_en   = qspi_valid && !fifo_full && (state == LOAD_FIFO);
        fifo_wr_data = qspi_data;
        qspi_ready   = !fifo_full && (state == LOAD_FIFO);
        fifo_rd_en   = (state == LOAD_FEATURES) && !fifo_empty
                       && (feat_wr_addr < FEATURE_DIM);
    end

    // =========================================================================
    // Feature Register Bank — Write Path (LOAD_FEATURES)
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            feat_wr_en_r   <= 1'b0;
            feat_wr_addr_r <= '0;
        end else begin
            feat_wr_en_r   <= fifo_rd_en;
            feat_wr_addr_r <= feat_wr_addr;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) feat_wr_addr <= '0;
        else begin
            case (state)
                LOAD_FEATURES: if (fifo_rd_en) feat_wr_addr <= feat_wr_addr + 1;
                default:       feat_wr_addr <= '0;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) feat_wr_count <= '0;
        else begin
            case (state)
                LOAD_FEATURES: if (feat_wr_en_r) feat_wr_count <= feat_wr_count + 1;
                default:       feat_wr_count <= '0;
            endcase
        end
    end

    always_ff @(posedge clk) begin
        if (feat_wr_en_r)
            feature_bank[feat_wr_addr_r] <= fifo_rd_data;
    end

    // =========================================================================
    // Feature Register Bank — Read Path (COMPUTE_DIST)
    // =========================================================================

    assign feat_rd_en = (state == COMPUTE_DIST) && (feat_rd_addr < FEATURE_DIM);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) feat_rd_en_r <= 1'b0;
        else        feat_rd_en_r <= feat_rd_en;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) feat_rd_addr <= '0;
        else begin
            case (state)
                COMPUTE_DIST: if (feat_rd_en) feat_rd_addr <= feat_rd_addr + 1;
                default:      feat_rd_addr <= '0;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) feat_rd_data <= '0;
        else if (feat_rd_en)
            feat_rd_data <= feature_bank[feat_rd_addr];
    end

    // =========================================================================
    // SV RAM Addressing
    // =========================================================================

    always_comb begin
        sv_base = {2'b00, sv_counter};
        if (class_counter >= 1) sv_base = sv_base + {2'b00, sv_count_reg[0]};
        if (class_counter >= 2) sv_base = sv_base + {2'b00, sv_count_reg[1]};
        if (class_counter >= 3) sv_base = sv_base + {2'b00, sv_count_reg[2]};
        if (class_counter >= 4) sv_base = sv_base + {2'b00, sv_count_reg[3]};
    end

    assign sv_ram_addr = {sv_base, feat_rd_addr};
    assign sv_ram_ren  = feat_rd_en;
    assign dist_sv_in  = sv_ram_rdata;

    // =========================================================================
    // Pipeline Connections
    // =========================================================================

    assign dist_start      = (state == COMPUTE_DIST);
    assign dist_feature_in = feat_rd_data;
    assign dist_valid_in   = feat_rd_en_r;

    assign horner_start    = (state == COMPUTE_KERNEL);
    assign horner_dist_in  = dist_out;
    assign horner_valid_in = dist_valid_latch;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            dist_valid_latch <= 1'b0;
        else if (dist_valid_out)
            dist_valid_latch <= 1'b1;
        else if (state == COMPUTE_KERNEL)
            dist_valid_latch <= 1'b0;
    end

    // =========================================================================
    // Counter Management
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_counter     <= '0;
            sv_counter         <= '0;
            class_counter      <= '0;
            gamma_latched      <= GAMMA_DEFAULT;
            num_samples_latched <= 10'd1;
            for (int i = 0; i < 5; i++) sv_count_reg[i] <= '0;
        end else begin
            case (state)
                IDLE: begin
                    sample_counter <= '0;
                    sv_counter     <= '0;
                    class_counter  <= '0;
                    if (start && vbatt_ok_s) begin
                        for (int i = 0; i < 5; i++)
                            sv_count_reg[i] <= num_sv_per_class[i];
                        // Latch gamma at the start of each batch so a mid-compute
                        // param_write_en cannot corrupt in-flight kernel values.
                        gamma_latched       <= gamma_int;
                        num_samples_latched <= num_samples;
                    end
                end
                LOAD_FIFO: begin
                    sv_counter    <= '0;
                    class_counter <= '0;
                end
                OUTPUT_RESULT: begin
                    if (kernel_ready && kernel_valid) begin
                        if (last_sv && last_class) begin
                            sv_counter     <= '0;
                            class_counter  <= '0;
                            sample_counter <= sample_counter + 1;
                        end else if (last_sv) begin
                            sv_counter    <= '0;
                            class_counter <= class_counter + 1;
                        end else begin
                            sv_counter <= sv_counter + 1;
                        end
                    end
                end
                default: begin end
            endcase
        end
    end

    // =========================================================================
    // Warm-up Counter + Interrupted Flag
    // =========================================================================
    // arm_interrupted only updates while rst_n=1.  It freezes as soon as rst_n
    // falls and holds the pre-reset value through the entire reset pulse.
    // When rst_n rises again, the interrupted block reads arm_interrupted at
    // the negedge-rst_n event, getting the correct pre-reset snapshot.
    // (Icarus non-standard NBA order: both heartbeat_count and interrupted in
    // one reset clause cause interrupted to read heartbeat_count=0 instead of
    // its pre-reset value; gating arm_interrupted bypasses that race entirely.)
    always_ff @(posedge clk)
        if (rst_n) arm_interrupted <= (heartbeat_count > 7'd0 && heartbeat_count < 7'd100);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            heartbeat_count <= 7'd0;
        else if ((state == OUTPUT_RESULT) && kernel_valid && kernel_ready
                 && last_sv && last_class && (heartbeat_count < 7'd100))
            heartbeat_count <= heartbeat_count + 7'd1;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            interrupted <= arm_interrupted; // reads pre-reset value (no NBA race)
        else if (heartbeat_count == 7'd100)
            interrupted <= 1'b0;
    end

    // =========================================================================
    // Work RAM Output
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) kernel_out_counter <= '0;
        else begin
            case (state)
                IDLE:          kernel_out_counter <= '0;
                OUTPUT_RESULT: if (kernel_ready && kernel_valid)
                                   kernel_out_counter <= kernel_out_counter + 1;
                default: ;
            endcase
        end
    end

    assign work_ram_addr  = kernel_out_counter;
    assign work_ram_wdata = kernel_out;
    assign work_ram_wen   = (state == OUTPUT_RESULT) && kernel_valid && kernel_ready;
    assign work_ram_ren   = 1'b0;

    // =========================================================================
    // Output Registers
    // =========================================================================

    // ── FIFO overflow sticky flag ─────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            fifo_overflow_r <= 1'b0;
        else if ((state == LOAD_FIFO) && fifo_full && qspi_valid)
            fifo_overflow_r <= 1'b1;
    end

    // ── Error priority encoder (combinational) ────────────────────────────────
    // Priority: illegal-state > SV-zero > SV-overflow > num-samples-zero >
    //           gamma-sat > gamma-zero > FIFO-overflow >
    //           power-fail > low-battery > warming-up (all advisory)
    always_comb begin
        if (state == ERROR_STATE)
            err_detect = ERR_ILLEGAL_STATE;
        else if ((state != IDLE) && (total_sv_check == 0))
            err_detect = ERR_SV_ZERO;
        else if ((state != IDLE) && (total_sv_check > NUM_SV))
            err_detect = ERR_SV_OVERFLOW;
        else if ((state != IDLE) && (num_samples == 0))
            err_detect = ERR_NUM_SAMPLES_ZERO;
        else if ((state != IDLE) && (gamma_int > GAMMA_SAT_THRESH))
            err_detect = ERR_GAMMA_SAT;
        else if ((state != IDLE) && (gamma_int == '0))
            err_detect = ERR_GAMMA_ZERO;
        else if (fifo_overflow_r)
            err_detect = ERR_FIFO_OVERFLOW;
        else if (!vbatt_ok_s)
            err_detect = ERR_POWER_FAIL;
        else if (vbatt_warn_s)
            err_detect = ERR_LOW_BATTERY;
        else if (heartbeat_count > 0 && heartbeat_count < 7'd100)
            err_detect = interrupted ? ERR_INTERRUPTED : ERR_WARMING_UP;
        else
            err_detect = ERR_NONE;
    end

    // ── Output registers ─────────────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done         <= 1'b0;
            error        <= 1'b0;
            error_code   <= ERR_NONE;
            kernel_out   <= '0;
            kernel_valid <= 1'b0;
        end else begin
            done  <= (state == OUTPUT_RESULT) && kernel_valid && kernel_ready
                      && last_sv && last_class && last_heartbeat;
            // Codes 0x1–0x7 are real faults: sticky (latch first, hold until rst_n).
            // Codes 0x8+ are advisory (ERR_WARMING_UP, ERR_INTERRUPTED): non-sticky,
            // auto-clear when err_detect returns ERR_NONE (heartbeat_count >= 100).
            if (err_detect != ERR_NONE && err_detect < 4'h8) begin
                // Real fault: latch first occurrence; overrides any advisory showing.
                if (error_code == ERR_NONE || error_code >= 4'h8) begin
                    error      <= 1'b1;
                    error_code <= err_detect;
                end
            end else if (err_detect >= 4'h8) begin
                // Advisory: show only while no real fault is latched.
                if (error_code == ERR_NONE || error_code >= 4'h8) begin
                    error      <= 1'b1;
                    error_code <= err_detect;
                end
            end else begin
                // ERR_NONE: clear advisory; real faults stay latched.
                if (error_code >= 4'h8) begin
                    error      <= 1'b0;
                    error_code <= ERR_NONE;
                end
            end
            // Latch kernel_out when Horner produces a new result; hold stable.
            if (horner_valid_out)
                kernel_out <= horner_kernel_out;
            // Hold kernel_valid high until the downstream consumer accepts it
            // (kernel_ready && kernel_valid).  Previously it dropped after 1
            // cycle because it just tracked horner_valid_out — this caused the
            // FSM to stall permanently if kernel_ready was low that cycle.
            if (horner_valid_out)
                kernel_valid <= 1'b1;
            else if (kernel_valid && kernel_ready)
                kernel_valid <= 1'b0;
        end
    end

endmodule

// ===========================================================================
// Input FIFO Module (16 KB SRAM)  — unchanged from ECE410_project_updated
// ===========================================================================

module input_fifo #(
    parameter int DATA_WIDTH = 16,
    parameter int DEPTH = 1024,
    parameter int ADDR_WIDTH = 10
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    wr_en,
    input  logic [DATA_WIDTH-1:0]   wr_data,
    input  logic                    rd_en,
    output logic [DATA_WIDTH-1:0]   rd_data,
    output logic                    full,
    output logic                    empty,
    output logic [ADDR_WIDTH:0]     count
);

    logic [DATA_WIDTH-1:0] mem [DEPTH];
    logic [ADDR_WIDTH:0]   wr_ptr;
    logic [ADDR_WIDTH:0]   rd_ptr;

    assign full  = (count == DEPTH);
    assign empty = (count == 0);

    wire [ADDR_WIDTH-1:0] wr_ptr_idx = wr_ptr[ADDR_WIDTH-1:0];
    wire [ADDR_WIDTH-1:0] rd_ptr_idx = rd_ptr[ADDR_WIDTH-1:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) wr_ptr <= '0;
        else if (wr_en && !full) wr_ptr <= wr_ptr + 1;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) rd_ptr <= '0;
        else if (rd_en && !empty) rd_ptr <= rd_ptr + 1;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= '0;
        end else begin
            case ({wr_en && !full, rd_en && !empty})
                2'b10:   count <= count + 1;
                2'b01:   count <= count - 1;
                default: count <= count;
            endcase
        end
    end

    always_ff @(posedge clk) begin
        if (wr_en && !full)
            mem[wr_ptr_idx] <= wr_data;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) rd_data <= '0;
        else if (rd_en && !empty) rd_data <= mem[rd_ptr_idx];
    end

endmodule

// ===========================================================================
// Distance Matrix Engine  — unchanged from ECE410_project_updated
// ===========================================================================

module distance_matrix #(
    parameter int DATA_WIDTH = 16,
    parameter int FRAC_BITS = 10,
    parameter int DIST_WIDTH = 20,
    parameter int FEATURE_DIM = 256
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    start,
    output logic                    done,
    input  logic [DATA_WIDTH-1:0]   feature_in,
    input  logic [DATA_WIDTH-1:0]   sv_in,
    input  logic                    valid_in,
    output logic [DIST_WIDTH-1:0]   dist_out,
    output logic                    valid_out
);

    typedef enum logic [1:0] {
        IDLE,
        ACCUMULATE,
        OUTPUT,
        DONE_STATE
    } state_t;

    state_t state, next_state;

    logic [2*DATA_WIDTH-1:0]   diff;
    logic [2*DATA_WIDTH-1:0]   diff_squared;
    logic [2*DATA_WIDTH+8-1:0] accumulator;
    logic [8:0]                dim_counter;
    // drain_cnt flushes the 2-cycle diff→diff_squared→accumulator pipeline after
    // the last valid_in so all FEATURE_DIM contributions are accumulated.
    //   drain_cnt=1: square the last diff (feature[255]-sv[255])
    //   drain_cnt=2: add that square to accumulator; then transition to OUTPUT
    logic [1:0]                drain_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            IDLE:       if (start) next_state = ACCUMULATE;
            ACCUMULATE: if (drain_cnt == 2'd2) next_state = OUTPUT;
            OUTPUT:     next_state = DONE_STATE;
            DONE_STATE: next_state = IDLE;
            default:    next_state = IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dim_counter <= '0;
        end else begin
            case (state)
                IDLE:       dim_counter <= '0;
                ACCUMULATE: if (valid_in) dim_counter <= dim_counter + 1;
                default:    dim_counter <= dim_counter;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            drain_cnt <= 2'd0;
        end else begin
            case (state)
                IDLE: drain_cnt <= 2'd0;
                ACCUMULATE: begin
                    if (dim_counter >= FEATURE_DIM - 1 && valid_in && drain_cnt == 2'd0)
                        drain_cnt <= 2'd1;
                    else if (drain_cnt != 2'd0)
                        drain_cnt <= drain_cnt + 2'd1;
                end
                default: drain_cnt <= 2'd0;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            diff         <= '0;
            diff_squared <= '0;
        end else if (state == IDLE) begin
            // Reset pipeline registers between SVs so stale values don't
            // contaminate the first accumulation of the next computation.
            diff         <= '0;
            diff_squared <= '0;
        end else if (state == ACCUMULATE) begin
            if (valid_in) begin
                diff         <= $signed(feature_in) - $signed(sv_in);
                diff_squared <= $signed(diff) * $signed(diff);
            end else if (drain_cnt == 2'd1) begin
                // Flush last diff through squaring stage
                diff_squared <= $signed(diff) * $signed(diff);
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= '0;
        end else begin
            case (state)
                IDLE:       accumulator <= '0;
                ACCUMULATE: if (valid_in || drain_cnt != 2'd0)
                    accumulator <= accumulator + (diff_squared >>> FRAC_BITS);
                default:    accumulator <= accumulator;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dist_out  <= {DIST_WIDTH{1'b0}};
            valid_out <= 1'b0;
            done      <= 1'b0;
        end else begin
            case (state)
                OUTPUT: begin
                    dist_out  <= (|accumulator[2*DATA_WIDTH+8-1:DIST_WIDTH])
                                  ? {DIST_WIDTH{1'b1}}
                                  : accumulator[DIST_WIDTH-1:0];
                    valid_out <= 1'b1;
                    done      <= 1'b0;
                end
                DONE_STATE: begin
                    valid_out <= 1'b0;
                    done      <= 1'b1;
                end
                default: begin
                    valid_out <= 1'b0;
                    done      <= 1'b0;
                end
            endcase
        end
    end

endmodule

// ===========================================================================
// Horner Engine  —  Range-Reduction LUT version
//
// Interface: identical to ECE410_project_updated horner_engine.
// Internals: replaces single-stage Horner with exp(-I)×exp(-F) decomposition.
//
// Latency: 18 cycles (SCALE + SCALE2 + HORNER_14..0 + OUTPUT = same as before)
//
// SCALE  : temp_p <= gamma_unsigned × dist_unsigned  (36-bit positive product P)
// SCALE2 : extract I = P>>20 → LUT lookup → lut_val
//          extract F_q = (P>>10)&0x3FF → x = -F_q  (x always ∈ [-1023,0])
// HORNER : standard 15th-order Horner on x → exp(-F) in Q6.10
// OUTPUT : kernel_out = (lut_val × horner_val) >> FRAC_BITS, clamped [0,1024]
// ===========================================================================

module horner_engine #(
    parameter int DATA_WIDTH = 16,
    parameter int FRAC_BITS = 10,
    parameter int DIST_WIDTH = 20
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    start,
    output logic                    done,
    input  logic [DATA_WIDTH-1:0]   gamma,
    input  logic [DIST_WIDTH-1:0]   dist_in,
    input  logic                    valid_in,
    output logic [DATA_WIDTH-1:0]   kernel_out,
    output logic                    valid_out
);

    // ── Horner polynomial coefficients (Q6.10, same as previous version) ─────
    localparam logic [DATA_WIDTH-1:0] COEFF_00 = (1 << FRAC_BITS);   // 1.0
    localparam logic [DATA_WIDTH-1:0] COEFF_01 = (1 << FRAC_BITS);
    localparam logic [DATA_WIDTH-1:0] COEFF_02 = (1 << (FRAC_BITS-1));
    localparam logic [DATA_WIDTH-1:0] COEFF_03 = ((1 << FRAC_BITS) / 6);
    localparam logic [DATA_WIDTH-1:0] COEFF_04 = ((1 << FRAC_BITS) / 24);
    localparam logic [DATA_WIDTH-1:0] COEFF_05 = ((1 << FRAC_BITS) / 120);
    localparam logic [DATA_WIDTH-1:0] COEFF_06 = ((1 << FRAC_BITS) / 720);
    localparam logic [DATA_WIDTH-1:0] COEFF_07 = ((1 << FRAC_BITS) / 5040);
    localparam logic [DATA_WIDTH-1:0] COEFF_08 = ((1 << FRAC_BITS) / 40320);
    localparam logic [DATA_WIDTH-1:0] COEFF_09 = ((1 << FRAC_BITS) / 362880);
    localparam logic [DATA_WIDTH-1:0] COEFF_10 = ((1 << FRAC_BITS) / 3628800);
    localparam logic [DATA_WIDTH-1:0] COEFF_11 = ((1 << FRAC_BITS) / 39916800);
    localparam logic [DATA_WIDTH-1:0] COEFF_12 = ((1 << FRAC_BITS) / 479001600);
    localparam logic [DATA_WIDTH-1:0] COEFF_13 = 1;
    localparam logic [DATA_WIDTH-1:0] COEFF_14 = 1;
    localparam logic [DATA_WIDTH-1:0] COEFF_15 = 1;

    // ── EXP_INT_LUT: exp(-i) × 1024 for i = 0..15, Q6.10 ────────────────────
    // [1024, 377, 139, 51, 19, 7, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    function automatic logic [DATA_WIDTH-1:0] exp_int_lut;
        input logic [3:0] idx;
        case (idx)
            4'd0:  return 16'd1024;
            4'd1:  return 16'd377;
            4'd2:  return 16'd139;
            4'd3:  return 16'd51;
            4'd4:  return 16'd19;
            4'd5:  return 16'd7;
            4'd6:  return 16'd3;
            4'd7:  return 16'd1;
            default: return 16'd0;   // exp(-8..15)×1024 < 0.5 → rounds to 0
        endcase
    endfunction

    typedef enum logic [4:0] {
        IDLE,
        SCALE,
        SCALE2,
        HORNER_14,
        HORNER_13,
        HORNER_12,
        HORNER_11,
        HORNER_10,
        HORNER_9,
        HORNER_8,
        HORNER_7,
        HORNER_6,
        HORNER_5,
        HORNER_4,
        HORNER_3,
        HORNER_2,
        HORNER_1,
        HORNER_0,
        OUTPUT
    } state_t;

    state_t state, next_state;

    // temp_p : 36-bit positive product P = gamma × dist_in (set in SCALE)
    // temp_h : 32-bit signed Horner multiply x × result (set in HORNER states)
    logic [DATA_WIDTH+DIST_WIDTH-1:0] temp_p;
    logic signed [2*DATA_WIDTH-1:0]   temp_h;
    logic signed [DATA_WIDTH-1:0]     x;
    logic [DATA_WIDTH-1:0]            lut_val;   // EXP_INT_LUT[I], set in SCALE2
    logic signed [DATA_WIDTH-1:0]     result;
    logic signed [DATA_WIDTH-1:0]     result_next;

    // Horner multiply shift: bits [25:10] of 32-bit temp_h → Q6.10 result
    // (Icarus workaround: parameterised bit-select extracted to wire)
    wire signed [DATA_WIDTH-1:0] temp_h_shifted;
    assign temp_h_shifted = temp_h[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS];

    // Combinational combination wires for OUTPUT state ─────────────────────
    // horner_clamp : clamp(result_next, 0, 1024) = exp(-F) in Q6.10
    // lut_product  : lut_val × horner_clamp (max = 1024 × 1024 = 2^20, 32-bit)
    // lut_product[25:10] = combined kernel output (max = 1024 = COEFF_00)
    wire [DATA_WIDTH-1:0]     horner_clamp;
    wire [2*DATA_WIDTH-1:0]   lut_product;

    assign horner_clamp = ($signed(result_next) < 0)                   ? '0       :
                          ($signed(result_next) > $signed(COEFF_00))   ? COEFF_00 :
                                                                          DATA_WIDTH'(result_next);
    assign lut_product  = lut_val * horner_clamp;

    // ── FSM ──────────────────────────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            IDLE:      if (start && valid_in) next_state = SCALE;
            SCALE:     next_state = SCALE2;
            SCALE2:    next_state = HORNER_14;
            HORNER_14: next_state = HORNER_13;
            HORNER_13: next_state = HORNER_12;
            HORNER_12: next_state = HORNER_11;
            HORNER_11: next_state = HORNER_10;
            HORNER_10: next_state = HORNER_9;
            HORNER_9:  next_state = HORNER_8;
            HORNER_8:  next_state = HORNER_7;
            HORNER_7:  next_state = HORNER_6;
            HORNER_6:  next_state = HORNER_5;
            HORNER_5:  next_state = HORNER_4;
            HORNER_4:  next_state = HORNER_3;
            HORNER_3:  next_state = HORNER_2;
            HORNER_2:  next_state = HORNER_1;
            HORNER_1:  next_state = HORNER_0;
            HORNER_0:  next_state = OUTPUT;
            OUTPUT:    next_state = IDLE;
            default:   next_state = IDLE;
        endcase
    end

    // ── result_next: forward the updated result so temp_h can use it immediately
    always_comb begin
        case (state)
            HORNER_14: result_next = COEFF_15;
            HORNER_13: result_next = $signed(COEFF_14) + $signed(temp_h_shifted);
            HORNER_12: result_next = $signed(COEFF_13) + $signed(temp_h_shifted);
            HORNER_11: result_next = $signed(COEFF_12) + $signed(temp_h_shifted);
            HORNER_10: result_next = $signed(COEFF_11) + $signed(temp_h_shifted);
            HORNER_9:  result_next = $signed(COEFF_10) + $signed(temp_h_shifted);
            HORNER_8:  result_next = $signed(COEFF_09) + $signed(temp_h_shifted);
            HORNER_7:  result_next = $signed(COEFF_08) + $signed(temp_h_shifted);
            HORNER_6:  result_next = $signed(COEFF_07) + $signed(temp_h_shifted);
            HORNER_5:  result_next = $signed(COEFF_06) + $signed(temp_h_shifted);
            HORNER_4:  result_next = $signed(COEFF_05) + $signed(temp_h_shifted);
            HORNER_3:  result_next = $signed(COEFF_04) + $signed(temp_h_shifted);
            HORNER_2:  result_next = $signed(COEFF_03) + $signed(temp_h_shifted);
            HORNER_1:  result_next = $signed(COEFF_02) + $signed(temp_h_shifted);
            HORNER_0:  result_next = $signed(COEFF_01) + $signed(temp_h_shifted);
            OUTPUT:    result_next = $signed(COEFF_00) + $signed(temp_h_shifted);
            default:   result_next = result;
        endcase
    end

    // ── Sequential datapath ───────────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x       <= '0;
            temp_p  <= '0;
            temp_h  <= '0;
            lut_val <= '0;
            result  <= '0;
        end else begin
            case (state)
                IDLE: result <= COEFF_00;

                // Compute positive product P = gamma × dist_in (36-bit unsigned)
                SCALE: begin
                    temp_p <= gamma * dist_in;
                end

                // Extract I (LUT index) and F_q (Horner input) from committed P.
                // I  = temp_p[35:20]  (P >> 20)
                // F_q = temp_p[19:10] ((P >> 10) & 0x3FF)
                // Overflow check: if temp_p[35:24] != 0 then I >= 16 → lut_val = 0.
                SCALE2: begin
                    x       <= -$signed({6'b0, temp_p[FRAC_BITS+9:FRAC_BITS]});
                    lut_val <= (|temp_p[DATA_WIDTH+DIST_WIDTH-1 : DIST_WIDTH+4])
                               ? '0
                               : exp_int_lut(temp_p[DIST_WIDTH+3 : DIST_WIDTH]);
                end

                HORNER_14: begin
                    temp_h <= $signed(x) * $signed(result_next);
                    result <= result_next;
                end
                HORNER_13: begin
                    temp_h <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_14) + $signed(temp_h[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_12: begin
                    temp_h <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_13) + $signed(temp_h[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_11: begin
                    temp_h <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_12) + $signed(temp_h[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_10: begin
                    temp_h <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_11) + $signed(temp_h[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_9: begin
                    temp_h <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_10) + $signed(temp_h[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_8: begin
                    temp_h <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_09) + $signed(temp_h[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_7: begin
                    temp_h <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_08) + $signed(temp_h[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_6: begin
                    temp_h <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_07) + $signed(temp_h[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_5: begin
                    temp_h <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_06) + $signed(temp_h[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_4: begin
                    temp_h <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_05) + $signed(temp_h[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_3: begin
                    temp_h <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_04) + $signed(temp_h[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_2: begin
                    temp_h <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_03) + $signed(temp_h[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_1: begin
                    temp_h <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_02) + $signed(temp_h[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                HORNER_0: begin
                    temp_h <= $signed(x) * $signed(result_next);
                    result <= $signed(COEFF_01) + $signed(temp_h[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                end
                default: begin end
            endcase
        end
    end

    // ── Output registers ─────────────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            kernel_out <= '0;
            valid_out  <= 1'b0;
            done       <= 1'b0;
        end else begin
            case (state)
                OUTPUT: begin
                    // result_next = COEFF_00 + temp_h_shifted = exp(-F) in Q6.10
                    // lut_product = lut_val × horner_clamp (32-bit)
                    // lut_product[25:10] = (exp(-I) × exp(-F)) >> 10 in Q6.10
                    kernel_out <= lut_product[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS];
                    valid_out  <= 1'b1;
                    done       <= 1'b1;
                end
                default: begin
                    valid_out <= 1'b0;
                    done      <= 1'b0;
                end
            endcase
        end
    end

endmodule

// ===========================================================================
// 2-FF Synchronizer  —  metastability barrier for async digital inputs
//
// RESET_VAL selects the reset-time state of the pipeline:
//   1'b1 for vbatt_ok  (assume power OK at POR)
//   1'b0 for vbatt_warn (no warning at POR)
// ===========================================================================

module sync_ff #(
    parameter int STAGES    = 2,
    parameter bit RESET_VAL = 1'b0
) (
    input  logic clk,
    input  logic rst_n,
    input  logic d,
    output logic q
);
    logic [STAGES-1:0] pipe;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) pipe <= {STAGES{RESET_VAL}};
        else        pipe <= {pipe[STAGES-2:0], d};
    end
    assign q = pipe[STAGES-1];
endmodule
