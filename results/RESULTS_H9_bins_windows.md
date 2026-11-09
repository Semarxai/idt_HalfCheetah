# H9 — sensitivity to bin count and window length
8 Nov 2026 · scripts halfcheetah/H9_bins_windows.py, H9_eval.py · outputs out/H9/, out/H9_summary.csv,
out/H9_point.txt, out/H9_boot_b*_W*.csv, out/H9_bootsum.txt

Grid: bins {3, 4, 5} × window W {250, 500, 1000}, stride 50, same z-scoring and reference as the published
pipeline. k-bin edges: equal-width over the reference range [lo, hi], recovered exactly from the published
3-bin edges (lo = 2e1 − e2, hi = 2e2 − e1); for k = 3 the published edges are used directly. Calibration:
windows lying wholly before step 15,000 (W 250: 296 windows; 500: 291; 1,000: 281); evaluation: windows
from 300. Per trial and configuration, max |z| over evaluation windows for P, H_f, H_b, ΔH (plug-in and
Chao–Shen) and windowed reward. Check: the (3 bins, W 500) cell reproduces the primary scores to 1e-13.
Nested protocol as H5 (ALPHA 0.25 primary); 200 bootstrap resamples per configuration with selection inside.

## Nested held-out detection (control alarms of 21), ALPHA 0.25

| bins | W | consensus CS | P CS | consensus plug-in | reward | consensus CS − reward [95%] |
|---|---|---|---|---|---|---|
| 3 | 250 | 64.3% (6) | 62.5% (6) | 60.7% (6) | 46.4% (6) | +17.9 [−0.0, +36.9] |
| **3** | **500** | **76.8% (6)** | **74.4% (6)** | 72.0% (8) | 50.6% (6) | **+26.2 [+11.3, +39.9]** |
| 3 | 1000 | 79.8% (7) | 80.4% (6) | 61.3% (9) | 57.1% (6) | +22.7 [+0.6, +36.3] |
| 4 | 250 | 70.8% (6) | 63.7% (6) | 51.8% (8) | 46.4% (6) | +24.4 [+3.5, +38.7] |
| 4 | 500 | 70.2% (6) | 58.3% (6) | 62.5% (6) | 50.6% (6) | +19.6 [+3.6, +40.6] |
| 4 | 1000 | 69.6% (8) | 66.1% (4) | 66.1% (5) | 57.1% (6) | +12.5 [−10.1, +34.5] |
| 5 | 250 | 61.9% (6) | 58.9% (6) | 35.7% (8) | 46.4% (6) | +15.5 [−10.8, +31.0] |
| 5 | 500 | 75.6% (5) | 67.9% (5) | 63.1% (10) | 50.6% (6) | +25.0 [+6.0, +37.5] |
| 5 | 1000 | 69.0% (11) | 69.0% (6) | 70.2% (6) | 57.1% (6) | +11.9 [−6.0, +32.8] |

P CS − reward [95%]: 3/250 [−8.4, +32.2] · 3/500 [+4.7, +40.5] · 3/1000 [+2.9, +38.1] · 4/250 [−6.0, +36.4] ·
4/500 [−11.9, +33.9] · 4/1000 [−10.1, +25.0] · 5/250 [−13.7, +25.0] · 5/500 [−6.0, +35.1] · 5/1000 [−14.4, +29.2].

## ALPHA 0.10
| bins | W | consensus CS | P CS | consensus plug-in | reward |
|---|---|---|---|---|---|
| 3 | 250 | 42.9% (6) | 45.2% (3) | 43.5% (3) | 28.0% (3) |
| 3 | 500 | 65.5% (5) | 64.3% (3) | 61.3% (2) | 35.1% (3) |
| 3 | 1000 | 65.5% (3) | 75.0% (3) | 58.9% (3) | 47.6% (3) |
| 4 | 250 | 62.5% (3) | 56.5% (3) | 31.0% (3) | 28.0% (3) |
| 4 | 500 | 50.0% (5) | 45.2% (3) | 51.8% (4) | 35.1% (3) |
| 4 | 1000 | 61.9% (4) | 57.7% (3) | 57.1% (4) | 47.6% (3) |
| 5 | 250 | 42.3% (3) | 46.4% (3) | 34.5% (3) | 28.0% (3) |
| 5 | 500 | 57.7% (3) | 50.6% (3) | 54.8% (4) | 35.1% (3) |
| 5 | 1000 | 64.3% (3) | 59.5% (3) | 63.1% (5) | 47.6% (3) |

## In-sample ±3σ (descriptive)
three-of-four plug-in / three-of-four Chao–Shen / reward, detection and control alarms:
3/250 67.3% 8 · 94.6% 18 · 59.5% 7 — 3/500 69.0% 5 · 86.9% 11 · 44.0% 5 — 3/1000 77.4% 11 · 86.3% 12 · 85.1% 16 —
4/250 68.5% 10 · 84.5% 15 · 59.5% 7 — 4/500 70.8% 11 · 95.8% 17 · 44.0% 5 — 4/1000 81.5% 14 · 91.7% 16 · 85.1% 16 —
5/250 67.3% 12 · 80.4% 12 · 59.5% 7 — 5/500 69.6% 11 · 86.3% 14 · 44.0% 5 — 5/1000 81.5% 14 · 92.9% 13 · 85.1% 16.

## Reading
- The telemetry consensus detects more than reward held-out in all nine configurations (+11.9 to +26.2
  points); the interval excludes zero in five, touches zero in one, and includes it in three (4/1000,
  5/250, 5/1000).
- The published configuration (3 bins, W 500) gives the largest and best-separated advantage. It was fixed
  before this analysis; the sweep shows the direction of the result does not depend on it, while the size
  does.
- Longer windows help reward (57.1% at W 1000) and, at ±3σ, inflate control alarms for every method;
  more bins with short windows (5/250) is the weakest cell, as expected from coverage (H1, G1).

## Implementation notes (superseded runs kept in out/_superseded/)
Three earlier H9 runs were replaced before any result was used: recomputed (not published) 3-bin edges;
numpy SD with ddof 0 instead of 1; NaN-propagating maximum. Convention used throughout (H2, H5, H9):
windows with zero interaction budget — the agent holding one state for a whole window, P undefined — are
skipped: 278 of 249,732 Chao–Shen windows (0.11%).
