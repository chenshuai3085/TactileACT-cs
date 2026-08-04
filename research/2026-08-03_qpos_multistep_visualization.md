# Qpos-conditioned multi-step ForeTac visualization

## Scope

The visualization uses existing 16-step qpos-conditioned foresight models.
Future robot states `qpos[t+1:t+16]` are treated as candidate joint
trajectories. No `actions/joint_abs` or DP-generated action is used in the core
mechanism calculation.

## Held-out data

- Board wiping: v8j episode 3 from the 260625 dataset.
- Vase wiping: v8j episode 4 from the 260630 dataset.
- Card swiping: v8j episode 0 from the 260707 success split.

All three episode paths occur in the corresponding checkpoint's saved
validation list.

## Multi-task forecast result

The selected contact-rich frames show the following mean marker-vector L2 over
the complete 16-step prediction horizon:

| Task | Frame | Mean marker L2 | H=16 marker L2 |
|---|---:|---:|---:|
| Board wiping | 178 | 0.789 px | 0.774 px |
| Vase wiping | 382 | 0.173 px | 0.178 px |
| Card swiping | 334 | 0.150 px | 0.147 px |

The larger v8j board error is consistent with the previously observed domain
shift between the board TacVAE/foresight training distribution and the 260625
v8j marker distribution.

## Qpos trajectory guidance result

For v8j board episode 3 at frame 306, a bounded gradient update was applied to
the future qpos chunk through the path:

`future qpos -> foresight -> predicted tactile -> fixed-action scorer branch`

The scorer action input is held fixed for both before/after evaluation, so the
reported score change is caused by the predicted tactile consequence rather
than an action-only scorer shortcut.

- Quality margin: -6.443 -> -2.921, gain +3.522.
- Joint-chunk L2 adjustment: 1.583 across 16 x 7 values.
- Per-value trust region: 0.03 to 0.30 degrees, scaled by scorer statistics.
- Mean change in the predicted marker sequence: 1.081 px.

Three contact stages are reported together rather than selecting only the
largest-gain frame:

| Stage | Frame | Margin before | Margin after | Gain | Joint-chunk L2 |
|---|---:|---:|---:|---:|---:|
| Contact | 204 | 5.359 | 6.893 | +1.534 | 1.574 |
| Stable interaction | 306 | -6.443 | -2.921 | +3.522 | 1.583 |
| Completion | 510 | -6.779 | -5.605 | +1.175 | 1.547 |

The contact frame is the strongest qualitative example because both before
and after margins are positive. Stable interaction and completion improve
according to the scorer but remain negative, so they should be described as
relative improvements rather than successful contact states.

This is a qpos-conditioned diagnostic. It does not establish the final
action-conditioned deployment claim.
