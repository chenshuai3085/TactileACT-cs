# 2026-06-21 Current Architecture and Recent Research Improvement Notes

## Current Project Position

The current system has three layers:

```text
1. DP action prior
   image + proprio + left tactile latent -> diffusion action chunk

2. Future contact predictor
   action chunk + current state/tactile -> future tactile consequence

3. TacQuality guidance
   predicted future contact -> quality score -> bounded gradient update on action
```

The active `260617-only` DP run is useful as the action prior / baseline.
It should not be framed as the main novelty by itself.  Recent work points to a
stronger story: future tactile/force consequences should be predicted and used
to guide diffusion-policy action generation at inference time.

## Relevant Recent Research Direction

Most relevant recent papers checked through arXiv API / arXiv pages:

| paper | arXiv | main implication |
|---|---|---|
| Inference-time Policy Steering via Vision and Touch | https://arxiv.org/abs/2606.14981 | supports tactile inference-time steering of generative policies |
| Dream-Tac | https://arxiv.org/abs/2606.08737 | supports action-conditioned future tactile/world dynamics |
| FAWAM | https://arxiv.org/abs/2606.08555 | force should be predicted and used for closed-loop correction, not only concatenated as observation |
| ContactWorld | https://arxiv.org/abs/2606.13877 | spatially structured and temporally continuous contact representations matter |
| Feedback World Model | https://arxiv.org/abs/2605.15705 | world-model guidance should be corrected by observed prediction error |
| PACT | https://arxiv.org/abs/2606.08414 | diffusion policies can be aligned with physical constraints via post-training / gradient-style updates |
| Fisher-Preserving Guidance | https://arxiv.org/abs/2605.29937 | guidance should be bounded/manifold-aware to avoid off-policy actions |
| Frequency-Aware Flow Matching | https://arxiv.org/abs/2606.20135 | action smoothness/frequency consistency is important for stable robot execution |

## What This Means for Our Architecture

The most defensible method story is:

```text
image/proprio/tactile DP prior
  -> sample action chunk
  -> multi-step tactile + force-aware Foresight predicts contact consequences
  -> TacQualityEnergy evaluates future contact quality
  -> bounded classifier/scorer gradient guidance edits the action
  -> server logs force/action/guidance traces for real rollout evaluation
```

This is not reranking. Reranking can remain a diagnostic baseline, but the core
method should be gradient guidance on the action sample.

## Highest-Value Improvements

1. Make board Foresight force-aware.

   Board quality is defined by force magnitude and force smoothness. Marker-only
   prediction is incomplete. The next useful predictor should output:

   ```text
   future marker latent / marker field
   future force proxy: |F|, Fz, delta |F|, delta Fz, torque magnitude, jerk proxy
   force-band logits: too_light / good / too_heavy / oscillatory
   contact gate logits
   ```

2. Use multi-horizon scoring.

   Board wiping is a trajectory-quality problem, not a single-frame problem.
   Score `t+1...t+16` by force band, contact continuity, temporal smoothness, and
   marker/force stability.

3. Keep contact-phase gating.

   Approach/reset should receive weak or zero wiping-quality guidance. Wiping
   contact frames should receive stronger force-band and smoothness guidance.

4. Keep trust-region / accept-only guidance.

   The serving code already has bounded updates and accept-only checks. This is
   important because unconstrained score maximization can push actions off the DP
   manifold.

5. Add prediction-error awareness after force-aware Foresight.

   If the predicted force/marker residual is high on recent real rollouts,
   guidance should be weakened. This matches the Feedback World Model direction.

## Current Evidence Boundary

What is supported now:

- The `260617-only` DP 2000-epoch run completed and checkpointed normally.
- The run strongly overfits after epoch 85 by held-out episode validation loss.
- The validation-selected rollout candidate remains:

  `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_best.pth`

- Server-side TacQuality guidance code supports final-action and denoising-step
  guidance, contact gating, and force-trace logging.
- Force-aware board Foresight training completed for 100 epochs:

  ```text
  run dir
    /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_forceaware_multistep16_boardvae_e100_bs16_0

  best checkpoint
    foresight_force_best.ckpt
    epoch 85
    best_val_total 0.2896

  last checkpoint
    foresight_force_last.ckpt
    epoch 99
    final val_total 0.3255
  ```

  The best checkpoint loaded successfully and should be used for downstream
  force-aware TacQualityEnergy experiments.  The last checkpoint is not the
  validation-selected model.

  Final validation diagnostics from the run summary:

  ```text
  force_proxy_mae 0.1612
  contact_acc     0.9091
  band_acc        0.9735
  band_bacc       0.9721
  ```

  This is still a consequence-model result only.  The guidance value still needs
  offline gradient audit and paired real rollout validation.

What is not proven yet:

- Real board wiping improvement from guidance.
- Force curve improvement versus baseline under paired real rollout conditions.
- That marker-only Foresight is enough for board force-quality guidance.

## Recommended Next Experiment

The force-aware Foresight run has completed.  The next useful experiment is not
more predictor training by default, but checking whether its score has a useful
gradient with respect to DP action chunks.

```bash
cd /home/chenshuai/Project/TactileACT-cs
# expected input checkpoint for the next audit
/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_forceaware_multistep16_boardvae_e100_bs16_0/foresight_force_best.ckpt
```

Next success criteria should include:

- gradient audit: finite gradient rate, score delta, action delta norm,
  trust-region pass rate
- offline DP action refinement: predicted quality improves without large action
  displacement
- real rollout comparison: baseline vs guided force traces and task result

Only after paired real rollout evidence should we claim that the scorer/guidance
improves actual board wiping behavior.

## 04:13 Verified Recent Arxiv Notes

I re-checked the recent-paper list with the arXiv API to avoid recording
uncertain titles as evidence.  The most relevant confirmed papers from the last
two months are:

| paper | arXiv | date | direct implication for this project |
|---|---:|---:|---|
| TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance | 2601.20239 | 2026-01-28 | Not within the last two months, but it remains the closest prior for inference-time tactile steering. Our difference should be outcome/quality-aware contact scoring rather than pure touch-action compatibility. |
| SI-Diff: A Framework for Learning Search and High-Precision Insertion with a Force-Domain Diffusion Policy | 2605.12247 | 2026-05-12 | Insertion benefits from force-domain modeling; supports keeping insertion and board quality heads physically grounded. |
| ForceFlow: Learning to Feel and Act via Contact-Driven Flow Matching | 2605.11048 | 2026-05-11 | Contact-rich policies should model force/contact evolution, not only image/proprio actions. |
| Tabero: Learning Gentle Manipulation with Closed-Loop Force Feedback from Vision, Touch, and Language | 2605.27886 | 2026-05-27 | Board wiping quality should explicitly include gentle/stable force feedback and closed-loop force traces. |
| Fisher-Preserving Guidance: Training-Free Manifold Constraints for Safe Diffusion Control | 2605.29937 | 2026-05-28 | Guidance updates need a trust region/manifold constraint; this supports our bounded/accept-only gradient updates. |
| PACT: Self-Evolving Physical Safety Alignment for Diffusion Policies in Embodied Manipulation | 2606.08414 | 2026-06-07 | Physical constraints can be enforced after pretraining; supports keeping DP as prior and adding a safety/quality score at inference/post-training time. |
| FAWAM: Force-Aware World Action Models for Closed-Loop Contact-Rich Manipulation | 2606.08555 | 2026-06-07 | Strong support for force-aware Foresight: force should appear in prediction and execution-time correction, not only as observation. |
| Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation | 2606.08737 | 2026-06-07 | Strong support for action-conditioned future tactile/world dynamics; this matches DP action chunk -> Foresight -> score. |
| TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation | 2606.11184 | 2026-06-09 | Very close recent prior for force-conditioned tactile latent prediction. It supports our force-aware Foresight design, but their predicted tactile latents are used as anticipatory policy features; our intended novelty should be converting predicted tactile/force consequences into a differentiable quality energy for DP action guidance. |
| ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation | 2606.13877 | 2026-06-11 | Supports evaluating representation properties and temporal contact continuity, not just single-step prediction loss. |
| Inference-time Policy Steering via Vision and Touch | 2606.14981 | 2026-06-12 | Directly supports the inference-time steering framing. |
| DREAM-Chunk: Reactive Action Chunking with Latent World Model | 2606.18589 | 2026-06-17 | Supports using a latent world model to correct action chunks at test time. |
| Frequency-Aware Flow Matching for Continuous and Consistent Robotic Action Generation | 2606.20135 | 2026-06-18 | Supports adding action/force smoothness and frequency consistency checks, especially for board wiping. |

Design consequence for our current codebase:

- Keep `diffusion/train_dp_tac_concat.py` as the action prior training path for now.
- Treat the `260617-only` 2000-epoch DP as a baseline/action-prior run, not the main novelty.
- The main research contribution should be a differentiable tactile/force
  consequence scorer:

  ```text
  action chunk from DP
    -> multi-step force-aware Foresight
    -> marker/force/contact-quality heads
    -> bounded classifier/scorer gradient guidance
    -> server-side force/action/guidance trace logging
  ```

- For board wiping, a marker-only future predictor is not enough because the
  positive/negative definition is force magnitude plus force smoothness during
  the contact wiping phase.
- `TacForeSight` is an important positioning reference.  It makes force-guided
  tactile foresight a timely and defensible direction, but it also means the
  paper story should not claim that force-conditioned tactile prediction itself
  is the main novelty.  The stronger distinction is:

  ```text
  recent tactile world models:
    force/tactile history -> future tactile latent -> policy feature

  our target method:
    DP action sample -> force-aware future tactile/force consequence
    -> differentiable contact-quality energy
    -> bounded action-gradient guidance
  ```

  This keeps the contribution on outcome-aware guidance for diffusion policies,
  not only on adding another tactile predictor.
- The completed high-value GPU job is the force-aware Foresight run:

  ```bash
  cd /home/chenshuai/Project/TactileACT-cs
  CONFIG=TFAC_V5/config_pretrain_foresight_board_forceaware_multistep16.json \
    scripts/train/train_foresight_board_forceaware_multistep16.sh
  ```

Evidence boundary:

- The above is a research/design conclusion from verified recent papers and the
  current codebase structure.
- It is not yet a real-robot performance claim.
- Real improvement must be judged by paired baseline-vs-guided board wiping
  rollouts with server-side force traces.

## 09:08 Force-Aware Foresight Gradient Audit

Added and ran a dedicated offline audit:

```text
TFAC_V5/tac_quality_energy/eval_force_aware_foresight_guidance.py
```

This audit uses the force-aware Foresight checkpoint:

```text
/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_forceaware_multistep16_boardvae_e100_bs16_0/foresight_force_best.ckpt
```

The score is built directly from force-aware Foresight outputs:

```text
quality_score =
  good-vs-risk force-band margin
  + contact log-probability
  - normalized force-center penalty
  - force smoothness penalty
```

This keeps the scorer differentiable with respect to the input action/state
chunk.  The trust-region update then performs bounded gradient ascent on the
score and only accepts updates that improve the predicted score.

Main held-out validation audit:

```text
output:
/home/chenshuai/Project/output/force_aware_foresight_guidance_audit/20260621_090725

split: val episodes
episodes: 31 held-out episodes from 301 total
samples: 372 contact-phase windows
seed: 42
refine_steps: 4
action_step: 0.02
max_total_delta: 0.08
```

Metrics:

```text
force-band acc           0.9795
force-band balanced acc  0.9736
contact acc              0.9207
good-vs-bad score AUC    1.0000
good-prob AUC            1.0000

finite grad rate         1.0000
positive grad rate       1.0000
improved rate            0.9409
trust-region pass        1.0000
score delta mean         3.5201
normalized action delta  0.0513
raw action delta norm    0.5848
```

Per-label base score behavior:

```text
good       mean score   6.6008, mean good_prob 9.4297e-01
too_small  mean score -19.7196, mean good_prob 9.3972e-08
too_large  mean score -13.8617, mean good_prob 4.6371e-06
oscillate  mean score -20.1418, mean good_prob 1.4555e-06
```

Repeat audits on the same episode-level val split and different random contact
windows remained stable:

```text
20260621_091044: seed 43, split_seed 42, band_bacc 0.9795, contact_acc 0.9096, improved_rate 0.9355
20260621_091049: seed 44, split_seed 42, band_bacc 0.9771, contact_acc 0.8901, improved_rate 0.9301
```

Interpretation:

- The force-aware Foresight scorer is a much better candidate for DP gradient
  guidance than a marker-only board scorer because board quality is explicitly
  defined by force magnitude and force smoothness during contact.
- The current result is a positive offline gradient audit: the score separates
  held-out good/bad windows and provides finite action gradients that improve
  the predicted quality under a small trust region.
- This still does not prove real-robot wiping improvement.  The next required
  evidence is paired baseline vs guided rollout with server-side force traces.

## 09:31 Force-Aware Board Serving Arm

The force-aware board scorer has now been connected as an optional serving arm:

```text
arm: force_aware_guided
runtime: ForceAwareForesightGuidanceRuntime
adapter: force_aware_foresight_trust_region_refinement
rollout config:
/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json
```

This arm is still not the default board recommendation.  The current default
for the planned board A/B remains `marker_joint_s12_guided`.  The force-aware
arm is a research arm for evaluating the stronger story:

```text
DP action chunk -> force-aware future tactile/force consequence
-> differentiable contact-quality energy -> bounded action-gradient guidance
```

Serving dry-run result:

```text
output:
/home/chenshuai/Project/output/tac_quality_guided_server_packet/board_force_aware_guided_smoke_20260621/guided_server_dry_run_smoke.json

dry_run_guidance_smoke_pass  true
runtime                      ForceAwareForesightGuidanceRuntime
adapter_policy               force_aware_foresight_trust_region_refinement
not_reranking                true
finite_grad_rate             1.0
positive_grad_rate           1.0
improved_rate                1.0
score_delta_mean             0.4879
raw_action_delta_mean        0.0746
```

The command sheet now contains block `2c` for launching this arm on port `8769`.
Use a separate rollout root for these trials:

```text
/home/chenshuai/Project/output/board_force_rollouts/260617_only_force_aware_scorer
```

Evidence boundary:

- This proves optional serving integration and dry-run gradient behavior.
- It still does not prove real board-wiping improvement.
- Real evidence requires paired baseline vs `force_aware_guided` rollouts with
  server-side `force_trace.csv` and task outcome metadata.

## 09:45 Force-Aware Real-HDF5-Window Serving Audit

The force-aware serving arm was additionally audited on real held-out HDF5
windows through the same serving helper and DP action normalizer used by the
server path.

Code:

```text
TFAC_V5/tac_quality_energy/audit_force_aware_serving_real_windows.py
```

Main output:

```text
/home/chenshuai/Project/output/force_aware_serving_real_window_audit/20260621_094429/force_aware_serving_real_window_audit.json
```

Setup:

```text
arm                 force_aware_guided
split               val
split_counts        301 all / 270 train / 31 val
windows             80
labels              16 each:
                    oscillate
                    positive_260617
                    positive_old
                    too_large
                    too_small
```

Metrics:

```text
pass                       true
finite_grad_rate_mean      1.0000
positive_grad_rate_mean    1.0000
improved_rate_mean         1.0000
accept_rate_mean           1.0000
trust_region_pass_rate     1.0000
score_delta_mean           1.8718
raw_action_delta_mean      0.0760
normalized_action_delta    0.0175
contact_metric_mean        3.9598
```

Interpretation:

- This is stronger than the synthetic serving smoke because it uses real board
  HDF5 windows and stratifies across all five current board labels.
- It verifies the deployment contract:

  ```text
  real marker/qpos window + dataset future-qpos action chunk
  -> build_serving_guidance_from_arm("board", "force_aware_guided")
  -> ForceAwareForesightGuidanceRuntime
  -> bounded trust-region action update
  ```

- It still does not prove online robot improvement because the input action
  chunks are replayed dataset future qpos chunks, not live DP rollouts executed
  on the robot.

Scorecard update:

```text
force_aware_board_real_window_serving_ready = true
```

## 10:05 Force-Aware Paired Real-Rollout Manifest

The real-rollout bookkeeping pipeline now supports the force-aware board
research arm separately from the current marker-joint-s12 default arm.

Manifest output:

```text
/home/chenshuai/Project/output/tac_quality_real_rollout_manifest/board_force_aware_manifest/tac_quality_rollout_manifest.csv
```

Planned comparison:

```text
task                 board only
pairs                3
baseline server      port 8765, arm baseline
guided server        port 8769, arm force_aware_guided
rollout root         /home/chenshuai/Project/output/board_force_rollouts/260617_only_force_aware_scorer
pair prefix          board_force_aware
```

Precheck:

```text
coverage json:
/home/chenshuai/Project/output/tac_quality_real_rollout_coverage/board_force_aware_coverage/tac_quality_real_rollout_coverage.json

status_counts        {"missing": 6}
board_ready          false
real evidence        false
```

Board-only unified eval precheck:

```text
/home/chenshuai/Project/output/tac_quality_real_rollout_eval/board_force_aware_tac_quality_precheck/tac_quality_real_rollout_eval.json
```

It correctly reports no board `force_trace.csv` yet and marks insertion as
`skipped` when `--skip_insertion` is used.

Interpretation:

- The force-aware arm now has a clean paired real-rollout protocol.
- It intentionally uses a separate rollout root from marker-joint-s12 to avoid
  mixing scorer variants in one evaluation.
- This is still setup/readiness only.  It does not add real robot evidence until
  the six planned trials produce non-synthetic server-side `force_trace.csv`.

## 10:15 Scorecard Manifest Readiness Wiring

The current scorecard now reads the force-aware board paired rollout artifacts
directly:

```text
manifest:
/home/chenshuai/Project/output/tac_quality_real_rollout_manifest/board_force_aware_manifest/tac_quality_rollout_manifest.json

coverage:
/home/chenshuai/Project/output/tac_quality_real_rollout_coverage/board_force_aware_coverage/tac_quality_real_rollout_coverage.json

precheck eval:
/home/chenshuai/Project/output/tac_quality_real_rollout_eval/board_force_aware_tac_quality_precheck/tac_quality_real_rollout_eval.json
```

The scorecard fields intentionally separate readiness from real evidence:

```text
force_aware_board_rollout_manifest_ready = true
force_aware_board_real_rollout_complete  = false
real_paired_rollout_complete             = false
goal_complete                            = false
```

This preserves the evidence boundary: the collection route is ready, but the
actual paired robot force traces are still missing.
