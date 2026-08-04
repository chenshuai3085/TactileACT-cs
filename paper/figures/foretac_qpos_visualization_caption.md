# ForeTac qpos-conditioned visualization captions

## Core mechanism

**Predicting multi-step tactile consequences before execution.** The top row
shows four stages from a held-out v8j board-wiping episode. At a stable-contact
frame, the recorded future joint-state chunk is treated as the candidate joint
trajectory. The qpos-conditioned foresight model predicts its tactile
consequence at horizons 1, 4, 8, 12, and 16. A bounded score-gradient update
changes the joint trajectory and produces a more stable predicted contact
sequence, increasing the contact-quality margin from -6.447 to -2.925.

## Multi-task prediction

**Multi-step tactile prediction on held-out real-robot episodes.** Each task
shows the current real-robot observation and tactile marker field, followed by
ground-truth and predicted marker fields at horizons 1, 4, 8, 12, and 16. The
right column reports marker-vector L2 error over the complete 16-step horizon.
The displayed examples come from held-out board-wiping, vase-wiping, and
card-swiping episodes.

## Multi-time guidance

**Predicted tactile consequences before and after guidance at multiple contact
stages.** Contact, stable interaction, and completion frames are expanded from
the same held-out v8j board-wiping episode. For every frame, the figure shows
the complete qpos-conditioned consequence at horizons 1, 4, 8, 12, and 16.
The quality margin improves from 5.369 to 6.900 at contact, from -6.447 to
-2.925 during stable interaction, and from -6.786 to -5.613 near completion.

## Evidence boundary

These figures use the existing qpos-conditioned checkpoints. The conditioning
trajectory is `qpos[t+1:t+16]`; `actions/joint_abs` and DP-generated actions are
not used. The figures are therefore suitable as qpos-conditioned diagnostics
and should be replaced or relabeled before making a formal action-conditioned
claim.
