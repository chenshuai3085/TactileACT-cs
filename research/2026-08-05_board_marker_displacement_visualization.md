# Board-wiping marker displacement visualization

Date: 2026-08-05

## Objective

Create reusable tactile marker displacement assets following the visual style
of Fig. 3 in Xue et al. (2025), *Reactive Diffusion Policy: Slow-Fast
Visual-Tactile Policy Learning for Contact-Rich Manipulation*.

## Method

The visualization uses the original left tactile RGB image and overlays the
recorded `9x9x2` marker displacement as yellow arrows. Each field is computed
as `marker[t] - marker[t_ref]`, where `t_ref=50` is a no-contact reference
frame from the same episode. This removes the sensor's static marker-offset
bias and matches the paper's `Flow(D0, Dt)` interpretation.

The 81 marker centers are detected directly from the reference tactile image.
As a consistency check on the normal wiping episode, detected image motion and
the stored marker field have x/y correlations of approximately 0.966/0.974.
Arrow lengths remain in the original pixel scale without magnification.

## Selected material

The sequence figure uses one normal wiping test episode and shows:

| Stage | Frame | Mean / max displacement |
|---|---:|---:|
| Approach / no contact | 50 | 0.00 / 0.00 px |
| Initial contact | 164 | 1.34 / 1.95 px |
| Stable wiping | 475 | 2.89 / 3.68 px |
| Peak shear | 588 | 5.68 / 6.92 px |

The contact-mode comparison uses explicit Board collection conditions:

| Mode | Frame | Mean / max displacement | Frame-to-frame change |
|---|---:|---:|---:|
| Insufficient pressure | 637 | 0.40 / 0.76 px | 0.365 px |
| Stable wiping | 475 | 2.89 / 3.68 px | 0.073 px |
| Excessive pressure | 530 | 7.87 / 10.24 px | 0.804 px |
| Slip / oscillation proxy | 642 | 3.89 / 4.57 px | 1.931 px |

The contrast is physically coherent: insufficient contact has negligible
deformation, excessive pressure has the largest absolute deformation, and the
oscillation proxy has a much larger temporal change than stable wiping.

## Artifacts

- Script: `scripts/visualize_board_marker_displacement.py`
- Sequence: `paper/figures/board_marker_displacement/board_marker_wiping_sequence.{png,pdf}`
- Modes: `paper/figures/board_marker_displacement/board_marker_contact_modes.{png,pdf}`
- Eight full-resolution reusable panels: `paper/figures/board_marker_displacement/panels/`
- Exact episode/frame provenance and metrics: `paper/figures/board_marker_displacement/metadata.json`
- Suggested captions: `paper/figures/board_marker_displacement/caption.md`

The low-pressure, excessive-pressure, and oscillation names are collection
condition proxies. They should not be presented as manually annotated
per-frame ground truth.
