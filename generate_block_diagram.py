"""
generate_block_diagram.py  v5
Saves block_diagram.png to ECE410_project_netlist.

Changes from v4:
  - vbatt_warn / vbatt_ok input pins added (async, 2-FF sync inside core)
  - FSM start guard updated to start && vbatt_ok_s
  - Error Codes box expanded: advisory codes 0x8-0xB added
  - Revisions box: v5 entry added
  - Distance matrix note updated to 256+2 drain cycles
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

C = {
    "host_face":    "#dcedf7",  "host_edge":    "#4a7fb5",
    "feat_face":    "#d8f0da",  "feat_edge":    "#4aad5a",
    "qspi_face":    "#ece6f5",  "qspi_edge":    "#8b7db5",
    "chip_face":    "#f8f8f8",  "chip_edge":    "#999999",
    "compute_face": "#fff5ef",  "compute_edge": "#d06030",
    "config_face":  "#f0eefa",  "config_edge":  "#9a8dc8",
    "mem_face":     "#fde8e8",  "mem_edge":     "#cc6060",
    "offchip_face": "#fff8e6",  "offchip_edge": "#c8a030",
    "inner_face":   "#ffffff",  "inner_edge":   "#aaaaaa",
    "adv_face":     "#f0faf0",  "adv_edge":     "#4aad5a",
    "tb":  "#2c5f8a",
    "to":  "#8a3010",
    "tr":  "#8a1010",
    "tg":  "#7a5a00",
    "tp":  "#4a3a80",
    "tgr": "#1a6e2a",
}

def bx(ax, x, y, w, h, face, edge, lw=1.4, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.02", facecolor=face, edgecolor=edge,
        linewidth=lw, linestyle=ls, transform=ax.transData, zorder=2))

def tx(ax, x, y, s, sz=11, color="#222", ha="center", va="center",
       weight="normal", style="normal"):
    ax.text(x, y, s, fontsize=sz, color=color, ha=ha, va=va,
            fontweight=weight, fontstyle=style, zorder=5,
            transform=ax.transData)

def seg(ax, xs, ys, color, lw=1.3):
    ax.plot(xs, ys, color=color, lw=lw, zorder=3)

def tip(ax, x0, y0, x1, y1, color, lw=1.3, shrink=2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                        connectionstyle="arc3,rad=0",
                        shrinkA=0, shrinkB=shrink), zorder=4)

# ════════════════════════════════════════════════════════════════════
#  PAGE 1  —  hardware block diagram
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(18, 22))
ax.set_xlim(0, 18); ax.set_ylim(0, 22)
ax.set_aspect("equal"); ax.axis("off")
fig.patch.set_facecolor("white")

# ── title ─────────────────────────────────────────────────────────
tx(ax, 9, 21.72, "Multi-Class Cardiac Arrhythmia Detection System",
   sz=16, weight="bold", color="#111")
tx(ax, 9, 21.35,
   "ECE410_project_LUT  ·  γ = 0.25  ·  LUT range-reduction kernel"
   "  ·  multi-scale 256-dim features  ·  causal 100-beat RR",
   sz=11, color="#555", style="italic")

# ── HOST MCU ──────────────────────────────────────────────────────
HX, HY, HW, HH = 0.25, 14.5, 6.0, 6.5
bx(ax, HX, HY, HW, HH, C["host_face"], C["host_edge"], lw=2.0)
tx(ax, HX+HW/2, HY+HH-0.30, "HOST MCU",
   sz=14, color=C["tb"], weight="bold")

bx(ax, HX+0.2, HY+5.0, 2.8, 0.75, C["inner_face"], C["inner_edge"], lw=1.0)
tx(ax, HX+0.2+1.4, HY+5.0+0.38,
   "R-peak detector  (MCU)", sz=11, color="#333")
tx(ax, HX+0.2+1.4, HY+5.0+0.75+0.28,
   "Raw ECG  (Lead II, 360 Hz)", sz=13, color="#555")

seg(ax, [HX+1.6, HX+1.6], [HY+5.0, HY+0.2+4.5], C["feat_edge"])
tip(ax, HX+1.6, HY+0.2+4.5+0.05, HX+1.6, HY+0.2+4.45, C["feat_edge"])

bx(ax, HX+0.2, HY+0.2, HW-0.4, 4.6, C["feat_face"], C["feat_edge"], lw=1.5)
tx(ax, HX+HW/2, HY+0.2+3.50,
   "Multi-scale Feature Extraction  (causal)", sz=12, color=C["tgr"], weight="bold")

tracks = [
    ("[0:128]",   "Single-beat morphology",
     "128 samples centred on R (±64 samp), amplitude-norm → [−1, 1]"),
    ("[128:192]", "10-beat mean morphology",
     "Mean of 10 × 64-sample snippets (prev. beats), amplitude-norm"),
    ("[192:256]", "100-beat RR track  (causal)",
     "99 RR intervals ÷ NORMAL_RR(308), clip [0,2], resample → 64 pts"),
]
for i, (dims, name, detail) in enumerate(tracks):
    ty = HY + 0.2 + 2.45 - i * 1.05
    bx(ax, HX+0.35, ty, HW-0.7, 0.90, C["inner_face"], C["inner_edge"], lw=0.8)
    tx(ax, HX+0.55, ty+0.65, dims,   sz=10, color=C["tgr"], ha="left", weight="bold")
    tx(ax, HX+0.55, ty+0.44, name,   sz=11, color="#222",   ha="left", weight="bold")
    tx(ax, HX+0.55, ty+0.22, detail, sz=10, color="#444",   ha="left")

tx(ax, HX+HW/2, HY+0.20+0.20,
   "Concatenate  →  256 × Q6.10  |  ~200 B timestamps + ~1.3 KB samples",
   sz=10, color="#333")

# ── QSPI  (left side, below HOST MCU) ────────────────────────────
QX, QY, QW, QH = 0.25, 12.55, 6.0, 2.0
bx(ax, QX, QY, QW, QH, C["qspi_face"], C["qspi_edge"], lw=1.5)
tx(ax, QX+QW/2, QY+QH-0.32, "QSPI Interface",
   sz=13, color="#4a3080", weight="bold")
tx(ax, QX+QW/2, QY+QH/2+0.05,
   "Mode 0  ·  4-lane  ·  4 MHz SCK", sz=11, color="#444")
tx(ax, QX+QW/2, QY+QH/2-0.38,
   "256 µs per heartbeat  @  2 MB/s", sz=11, color="#444")

tip(ax, HX+HW/2, HY, HX+HW/2, QY+QH, C["feat_edge"])

# ── CHIP BOUNDARY ─────────────────────────────────────────────────
CBX, CBY, CBW, CBH = 6.7, 3.8, 11.0, 17.2
bx(ax, CBX, CBY, CBW, CBH, C["chip_face"], C["chip_edge"], lw=2.0, ls="--")
tx(ax, CBX+CBW/2, CBY+0.25, "CHIP BOUNDARY",
   sz=11, color="#888", weight="bold")

# ── PARAMETER REGISTERS ───────────────────────────────────────────
PRX, PRY, PRW, PRH = 7.2, 19.0, 10.0, 1.80
bx(ax, PRX, PRY, PRW, PRH, C["config_face"], C["config_edge"], lw=1.5)
tx(ax, PRX+PRW/2, PRY+PRH-0.30,
   "Parameter Registers", sz=12, color=C["tp"], weight="bold")
tx(ax, PRX+PRW/2, PRY+PRH/2-0.15,
   "gamma_reg 0x0100=0.25 (Q6.10)  |  c_reg 0x0400=1.0  |  "
   "bias_reg[0..4] Q6.10×5  |  sv_count[0..4] 8-bit×5",
   sz=10, color="#333")
tx(ax, PRX+PRW/2, PRY+0.30,
   "param_addr[2:0]  3-bit register select  |  latched on start pulse",
   sz=10, color="#555")

PR_CY = PRY + PRH / 2
seg(ax, [HX+HW, CBX, CBX+0.02], [PR_CY, PR_CY, PR_CY], C["config_edge"], lw=1.2)
tip(ax, CBX, PR_CY, PRX, PR_CY, C["config_edge"], lw=1.3)
tx(ax, (HX+HW + CBX)/2, PR_CY+0.28,
   "param_write_en / param_addr[2:0] / param_data[15:0]",
   sz=10, color=C["tp"])

# ── INPUT FIFO ────────────────────────────────────────────────────
FIX, FIY, FIW, FIH = 7.2, 16.8, 5.7, 2.2
bx(ax, FIX, FIY, FIW, FIH, C["mem_face"], C["mem_edge"], lw=1.5)
tx(ax, FIX+FIW/2, FIY+FIH-0.30,
   "Input FIFO  (ON-CHIP SRAM)", sz=12, color=C["tr"], weight="bold")
tx(ax, FIX+FIW/2, FIY+FIH/2-0.05,
   "8192 × 16-bit  =  16 KB  (32 heartbeats deep)", sz=11, color="#333")
tx(ax, FIX+FIW/2, FIY+0.35,
   "full / empty / count[13:0]", sz=10, color="#555")

QR_Y = QY + QH / 2
FI_Y = FIY + FIH / 2
seg(ax, [QX+QW, 6.85, 6.85], [QR_Y, QR_Y, FI_Y], C["qspi_edge"], lw=1.4)
tip(ax, 6.85, FI_Y, FIX, FI_Y, C["qspi_edge"], lw=1.4)
tx(ax, 6.6, (QR_Y+FI_Y)/2, "qspi_valid\ndata / ready",
   sz=10, color=C["qspi_edge"], ha="right")

# ── FEATURE BANK ──────────────────────────────────────────────────
FBX, FBY, FBW, FBH = 7.2, 14.3, 5.7, 2.2
bx(ax, FBX, FBY, FBW, FBH, C["feat_face"], C["feat_edge"], lw=1.5)
tx(ax, FBX+FBW/2, FBY+FBH-0.30,
   "Feature Bank  (ON-CHIP REGs)", sz=12, color=C["tgr"], weight="bold")
tx(ax, FBX+FBW/2, FBY+FBH/2-0.05,
   "256 × 16-bit  =  512 B", sz=11, color="#333")
tx(ax, FBX+FBW/2, FBY+0.35,
   "Written once per heartbeat; re-read × N_SV", sz=10, color="#555")

FI_CX = FIX + FIW/2
tip(ax, FI_CX, FIY, FI_CX, FBY+FBH, C["feat_edge"], lw=1.4)
tx(ax, FI_CX+0.12, (FIY+FBY+FBH)/2,
   "LOAD_FEATURES", sz=11, color=C["tgr"], ha="left")

# ── DISTANCE MATRIX ENGINE ────────────────────────────────────────
DMX, DMY, DMW, DMH = 7.2, 11.8, 7.0, 2.2
bx(ax, DMX, DMY, DMW, DMH, C["compute_face"], C["compute_edge"],
   lw=2.0, ls="--")
tx(ax, DMX+DMW/2, DMY+DMH-0.30,
   "Distance Matrix Engine", sz=12, color=C["to"], weight="bold")
tx(ax, DMX+DMW/2, DMY+DMH-0.72,
   "D  =  ‖ x[i] − sv[j] ‖²", sz=12, color="#333", style="italic")
tx(ax, DMX+DMW/2, DMY+DMH-1.10,
   "256 iterations per SV  |  +2 drain cycles  |  dist_out: 20-bit Q6.10",
   sz=10, color="#555")
tx(ax, DMX+DMW/2, DMY+0.32,
   "← sv_in  one 16-bit feature word per cycle  (from SV RAM)",
   sz=10, color="#888")

FB_CX = FBX + FBW/2
tip(ax, FB_CX, FBY, FB_CX, DMY+DMH, C["compute_edge"], lw=1.4)
tx(ax, FB_CX+0.12, (FBY+DMY+DMH)/2,
   "feature_in", sz=11, color=C["to"], ha="left")

# ── LUT RANGE-REDUCTION HORNER ENGINE ────────────────────────────
HEX, HEY, HEW, HEH = 7.2, 7.9, 7.0, 3.6
bx(ax, HEX, HEY, HEW, HEH, C["compute_face"], C["compute_edge"],
   lw=2.0, ls="--")
tx(ax, HEX+HEW/2, HEY+HEH-0.30,
   "LUT Range-Reduction Horner Engine",
   sz=12, color=C["to"], weight="bold")

stages = [
    ("SCALE",   "P = γ × D  →  36-bit positive product"),
    ("SCALE2",  "I=P>>20  |  F=(P>>10)&0x3FF  |  x=−F ∈ [−1023, 0]"),
    ("LUT",     "lut_val=EXP_INT_LUT[I]   [1024, 377, 139, 51, 19, 7, 3, 1, 0×8]"),
    ("HORNER",  "Horner 15th-order on x  →  exp(−F)   coeff ROM 32 B   18 cycles"),
    ("OUTPUT",  "kernel=(lut×horner)>>10   clamp[0,1024]  Q6.10  max|err|=0.00055"),
]
s_y0 = HEY + HEH - 0.65
s_h  = 0.50
for i, (label, detail) in enumerate(stages):
    sy = s_y0 - i * (s_h + 0.06)
    bx(ax, HEX+0.15, sy-s_h, HEW-0.30, s_h,
       C["inner_face"], C["inner_edge"], lw=0.8)
    tx(ax, HEX+0.42, sy-s_h/2, label,
       sz=10, color=C["to"], weight="bold", ha="left")
    tx(ax, HEX+1.10, sy-s_h/2, detail,
       sz=9, color="#333", ha="left")

DM_CX = DMX + DMW/2
tip(ax, DM_CX, DMY, DM_CX, HEY+HEH, C["compute_edge"], lw=1.4)
tx(ax, DM_CX+0.12, (DMY+HEY+HEH)/2,
   "dist_out (20-bit)", sz=11, color=C["to"], ha="left")

PR_RX = PRX + PRW
HE_RX = HEX + HEW
HE_CY = HEY + HEH/2
seg(ax, [PR_RX-1.5, 16.9, 16.9], [PRY, PRY, HE_CY], C["config_edge"], lw=1.3)
tip(ax, 16.9, HE_CY, HE_RX, HE_CY, C["config_edge"], lw=1.3)
tx(ax, 17.0, (PRY+HE_CY)/2, "γ  (Q6.10)",
   sz=11, color=C["tp"], ha="left")

# ── FSM CONTROLLER ────────────────────────────────────────────────
FCX, FCY, FCW, FCH = 7.2, 5.6, 10.0, 2.0
bx(ax, FCX, FCY, FCW, FCH, C["compute_face"], C["compute_edge"],
   lw=2.0, ls="--")
tx(ax, FCX+FCW/2, FCY+FCH-0.30,
   "FSM Controller", sz=12, color=C["to"], weight="bold")
tx(ax, FCX+FCW/2, FCY+FCH-0.68,
   "IDLE (start && vbatt_ok_s) → LOAD_FIFO → LOAD_FEATURES → "
   "COMPUTE_DIST (258 cyc) → COMPUTE_KERNEL (18 cyc) → OUTPUT_RESULT",
   sz=10, color="#333")
tx(ax, FCX+FCW/2, FCY+0.35,
   "Counters: sample / sv / class / feat_wr / feat_rd  "
   " |  Validation: 0 < Σsv_count ≤ NUM_SV  |  Sticky faults (0x1-7) + Advisory (0x8-B)",
   sz=10, color="#555")

HE_CX = HEX + HEW/2
tip(ax, HE_CX, HEY, HE_CX, FCY+FCH, C["tb"], lw=1.4)
tx(ax, HE_CX+0.12, (HEY+FCY+FCH)/2,
   "kernel_out / kernel_valid (Q6.10)", sz=11, color=C["tb"], ha="left")

_de_y, _de_cx = FCY+1.5, 6.56
seg(ax, [FCX, CBX, _de_cx, _de_cx], [_de_y, _de_y, _de_y, 17.5], "#444", lw=1.3)
tip(ax, _de_cx, 17.5, HX+HW, 17.5, "#444", lw=1.3)
tx(ax, CBX-0.10, _de_y+0.25, "done / error", sz=10, color="#444", ha="right")

# ── vbatt_warn / vbatt_ok input pins ─────────────────────────────
# Route from HOST MCU right edge → chip boundary → FSM bottom area
_vb_y = FCY + 0.55
seg(ax, [HX+HW, CBX, FCX], [_vb_y, _vb_y, _vb_y], "#2a7a3a", lw=1.2)
tip(ax, FCX+0.01, _vb_y, FCX+0.02, _vb_y, "#2a7a3a", lw=1.2)
tx(ax, CBX-0.12, _vb_y+0.22,
   "vbatt_warn / vbatt_ok\n(async → 2-FF sync_ff)", sz=9, color="#2a7a3a", ha="right")

# ── SV RAM ────────────────────────────────────────────────────────
SVX, SVY, SVW, SVH = 7.2, 1.2, 4.3, 2.35
bx(ax, SVX, SVY, SVW, SVH, C["offchip_face"], C["offchip_edge"], lw=1.5)
tx(ax, SVX+SVW/2, SVY+SVH-0.30,
   "SV RAM  (OFF-CHIP)", sz=12, color=C["tg"], weight="bold")
tx(ax, SVX+SVW/2, SVY+SVH/2-0.05,
   "477 SVs × 256 feat × 2 B  ≈ 244 KB\n"
   "18-bit addr  |  Read-only  |  Loaded at boot",
   sz=10, color="#333")

# ── WORKSPACE RAM ─────────────────────────────────────────────────
WRX, WRY, WRW, WRH = 12.3, 1.2, 4.5, 2.35
bx(ax, WRX, WRY, WRW, WRH, C["offchip_face"], C["offchip_edge"], lw=1.5)
tx(ax, WRX+WRW/2, WRY+WRH-0.30,
   "Workspace RAM  (OFF-CHIP)", sz=12, color=C["tg"], weight="bold")
tx(ax, WRX+WRW/2, WRY+WRH/2-0.05,
   "1000 × 250 × 2 B  ≤ 500 KB\n"
   "19-bit addr  |  Read / Write",
   sz=10, color="#333")

SV_CX = SVX + SVW/2
tip(ax, SV_CX, FCY, SV_CX, SVY+SVH, C["offchip_edge"], lw=1.3)
tx(ax, SV_CX+0.12, (FCY + SVY+SVH)/2,
   "sv_ram_addr[17:0]\nsv_ram_ren", sz=10, color=C["tg"], ha="left")

WR_CX = WRX + WRW/2
tip(ax, WR_CX, FCY, WR_CX, WRY+WRH, C["offchip_edge"], lw=1.3)
tx(ax, WR_CX+0.12, (FCY + WRY+WRH)/2,
   "work_ram_addr[18:0]\nwork_ram_wen/ren", sz=10, color=C["tg"], ha="left")

_wr_cx = 6.44
seg(ax, [WR_CX, _wr_cx, _wr_cx], [WRY+WRH, WRY+WRH, 16.0], C["offchip_edge"], lw=1.2)
tip(ax, _wr_cx, 16.0, HX+HW, 16.0, C["offchip_edge"], lw=1.2)
tx(ax, _wr_cx-0.08, 9.5, "kernel results\n→ HOST (OvO classify)", sz=9,
   color=C["tg"], ha="right")

DM_RX  = DMX + DMW
DM_CY  = DMY + DMH/2
SV_RX  = SVX + SVW
SV_TY  = SVY + SVH
seg(ax, [SV_RX, 17.1, 17.1], [SV_TY, SV_TY, DM_CY], C["offchip_edge"], lw=1.3)
tip(ax, 17.1, DM_CY, DM_RX, DM_CY, C["offchip_edge"], lw=1.3)
tx(ax, 17.15, (SV_TY+DM_CY)/2, "sv_in\n(16-bit,\none feat/cyc)",
   sz=10, color=C["tg"], ha="left")

# ════════════════════════════════════════════════════════════════════
#  LEFT COLUMN  —  Arrhythmia Classes, Legend, Error Codes, Revisions
#  All four boxes sized so revision bottom = SV RAM bottom (y=1.20)
# ════════════════════════════════════════════════════════════════════
LX, LW = 0.25, 6.05
GAP     = 0.15
REV_BOT = SVY           # 1.20  — align with SV RAM bottom
ACY_TOP = QY            # 12.55 — start just below QSPI

ACY_H   = 1.72          # Arrhythmia Classes
LEG_H   = 2.00          # Legend
ERR_H   = 5.15          # Error Codes (single box, no sub-box)
# Derive REV_H from the remaining space
REV_H   = (ACY_TOP - REV_BOT) - ACY_H - LEG_H - ERR_H - 3*GAP  # ≈ 1.53

LEG_TOP = ACY_TOP - ACY_H - GAP
ERR_TOP = LEG_TOP - LEG_H - GAP
REV_TOP = ERR_TOP - ERR_H - GAP

# ── Arrhythmia Classes ────────────────────────────────────────────
bx(ax, LX, ACY_TOP-ACY_H, LW, ACY_H, "#f5f5ff", "#7a6ab5", lw=1.3)
tx(ax, LX+LW/2, ACY_TOP-0.28,
   "Arrhythmia Classes  (5-class output)", sz=11, color="#4a3a80", weight="bold")
classes = [
    ("N",   "Normal sinus rhythm"),
    ("PVC", "Premature Ventricular Contraction"),
    ("AFib","Atrial Fibrillation"),
    ("VT",  "Ventricular Tachycardia"),
    ("SVT", "Supraventricular Tachycardia"),
]
for i, (sym, name) in enumerate(classes):
    cy = ACY_TOP - 0.62 - i*0.24
    tx(ax, LX+0.40, cy, sym,  sz=10, color="#4a3a80", weight="bold", ha="left")
    tx(ax, LX+0.90, cy, f"—  {name}", sz=10, color="#333", ha="left")

# ── Legend (compact, 2 columns × 4 rows) ─────────────────────────
bx(ax, LX, LEG_TOP-LEG_H, LW, LEG_H, "#fafafa", "#cccccc", lw=1.0)
tx(ax, LX+LW/2, LEG_TOP-0.25, "Legend", sz=11, color="#333", weight="bold")
legend_items = [
    (C["host_face"],    C["host_edge"],    "-",  "Host MCU"),
    (C["feat_face"],    C["feat_edge"],    "-",  "Feature extract/bank"),
    (C["qspi_face"],    C["qspi_edge"],    "-",  "QSPI interface"),
    (C["mem_face"],     C["mem_edge"],     "-",  "On-chip SRAM"),
    (C["compute_face"], C["compute_edge"], "--", "Compute engine (RTL)"),
    (C["config_face"],  C["config_edge"],  "-",  "Config registers"),
    (C["offchip_face"], C["offchip_edge"], "-",  "Off-chip RAM"),
    (C["chip_face"],    C["chip_edge"],    "--", "Chip boundary"),
]
row_h_leg = (LEG_H - 0.40) / 4
for i, (face, edge, ls, label) in enumerate(legend_items):
    lx2 = LX + 0.22 + (i % 2) * 2.95
    ly  = LEG_TOP - 0.50 - (i // 2) * row_h_leg
    ax.add_patch(mpatches.Rectangle((lx2, ly-0.10), 0.26, 0.22,
        facecolor=face, edgecolor=edge, linewidth=1.2, linestyle=ls, zorder=5))
    tx(ax, lx2+0.38, ly+0.01, label, sz=9.5, color="#333", ha="left")

# ── Error Codes (single box, sticky + advisory, no sub-box) ──────
bx(ax, LX, ERR_TOP-ERR_H, LW, ERR_H, "#fef9f9", "#cc6060", lw=1.3)
tx(ax, LX+LW/2, ERR_TOP-0.25,
   "Error Codes  (error_code[3:0])", sz=11, color=C["tr"], weight="bold")

# Column headers
tx(ax, LX+0.35, ERR_TOP-0.52, "Code", sz=9, color="#555", weight="bold", ha="left")
tx(ax, LX+1.05, ERR_TOP-0.52, "Name", sz=9, color="#555", weight="bold", ha="left")
tx(ax, LX+3.35, ERR_TOP-0.52, "Trigger", sz=9, color="#555", weight="bold", ha="left")
ax.plot([LX+0.22, LX+LW-0.22], [ERR_TOP-0.65, ERR_TOP-0.65],
        color="#cc9999", lw=0.8, zorder=3)

all_codes = [
    ("0x0", "ERR_NONE",             "No fault",               "sticky", True),
    ("0x1", "ERR_SV_ZERO",          "Σsv_count = 0",          "sticky", False),
    ("0x2", "ERR_SV_OVERFLOW",      "Σsv_count > NUM_SV",     "sticky", True),
    ("0x3", "ERR_ILLEGAL_STATE",    "FSM default branch",     "sticky", False),
    ("0x4", "ERR_GAMMA_SAT",        "gamma_int > 8192",       "sticky", True),
    ("0x5", "ERR_FIFO_OVERFLOW",    "QSPI word dropped",      "sticky", False),
    ("0x6", "ERR_GAMMA_ZERO",       "gamma_int = 0",          "sticky", True),
    ("0x7", "ERR_NUM_SAMPLES_ZERO", "num_samples = 0",        "sticky", False),
    None,   # separator
    ("0x8", "ERR_WARMING_UP",       "Cold start <100 beats",  "adv",    True),
    ("0x9", "ERR_INTERRUPTED",      "Reset mid-warmup",       "adv",    False),
    ("0xA", "ERR_LOW_BATTERY",      "vbatt_warn asserted",    "adv",    True),
    ("0xB", "ERR_POWER_FAIL",       "vbatt_ok deasserted",    "adv",    False),
]

row_h_err = (ERR_H - 0.72 - 0.30) / (len(all_codes))   # ~0.33
ey0 = ERR_TOP - 0.72
row_cursor = 0
for item in all_codes:
    if item is None:
        # Separator line + advisory label
        sep_y = ey0 - row_cursor * row_h_err
        ax.plot([LX+0.22, LX+LW-0.22], [sep_y+0.05, sep_y+0.05],
                color="#4aad5a", lw=0.8, ls="--", zorder=3)
        tx(ax, LX+0.35, sep_y - 0.08,
           "Advisory — auto-clear, non-sticky  (error_code ≥ 0x8)",
           sz=8, color=C["tgr"], ha="left", style="italic")
        row_cursor += 1
        continue
    code, name, note, kind, shade = item
    ey = ey0 - row_cursor * row_h_err
    if shade:
        ax.add_patch(mpatches.Rectangle(
            (LX+0.10, ey - row_h_err*0.5 + 0.02), LW-0.20, row_h_err - 0.04,
            facecolor="#f5eeee", edgecolor="none", zorder=1))
    code_color = C["tr"] if kind == "sticky" else C["tgr"]
    tx(ax, LX+0.35, ey, code, sz=9, color=code_color, weight="bold", ha="left")
    tx(ax, LX+1.05, ey, name, sz=9, color="#222", ha="left")
    tx(ax, LX+3.35, ey, note, sz=8, color="#555", ha="left")
    row_cursor += 1

# ── Revisions (compact, single line per entry) ────────────────────
bx(ax, LX, REV_BOT, LW, REV_H, "#f5f5f5", "#999999", lw=1.2)
tx(ax, LX+0.35, REV_TOP-0.25, "Revisions",
   sz=11, color="#333", weight="bold", ha="left")
ax.plot([LX+0.22, LX+LW-0.22], [REV_TOP-0.40, REV_TOP-0.40],
        color="#bbbbbb", lw=0.7, zorder=3)

revisions = [
    ("v5", "2026-05-18", "fixes 4-10 (power pins, sync_ff, drain flush)  22/22 PASS"),
    ("v4", "2026-05-05", "fixes 1-3 (kernel_valid, gamma shadow, ERR_GAMMA_ZERO)"),
    ("v3", "—",          "LUT range-reduction kernel, 256-dim multi-scale features"),
    ("v2", "—",          "20-bit accumulator, Horner polynomial"),
    ("v1", "—",          "Initial 10-bit fixed-point SVM core"),
]
n_rev = len(revisions)
row_h_rev = (REV_H - 0.48) / n_rev
for i, (ver, date, desc) in enumerate(revisions):
    ry = REV_TOP - 0.52 - i * row_h_rev
    tx(ax, LX+0.35, ry, ver,  sz=9, color="#333", weight="bold", ha="left")
    tx(ax, LX+0.75, ry, date, sz=8, color="#888", ha="left")
    tx(ax, LX+1.55, ry, desc, sz=8, color="#444", ha="left")

# Page footer
tx(ax, 9, 0.12, "Page 1 of 1  —  Hardware blocks  (v5)", sz=10, color="#aaa")

plt.tight_layout(pad=0.4)
out = os.path.join(OUT_DIR, "block_diagram.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
plt.close(fig)
